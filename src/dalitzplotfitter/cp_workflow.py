"""High-level workflow for simultaneous direct-CP Dalitz fits."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.background import CPBackgroundCategory
from dalitzplotfitter.constraints import ConstrainedNLL
from dalitzplotfitter.fit import Minimizer, Parameter
from dalitzplotfitter.io import read_phase_space_sample
from dalitzplotfitter.kinematics import PhaseSpaceSample
from dalitzplotfitter.likelihood import CPJointNLL
from dalitzplotfitter.plotting import plot_binned_data


def _collect_parameters(value: object) -> tuple[Parameter, ...]:
    if isinstance(value, Parameter):
        return (value,)
    if value is None:
        return ()
    parameters = getattr(value, "parameters", None)
    if parameters is not None and not callable(parameters):
        try:
            return tuple(item for item in parameters if isinstance(item, Parameter))
        except TypeError:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        found: list[Parameter] = []
        for field in fields(value):
            found.extend(_collect_parameters(getattr(value, field.name)))
        return tuple(found)
    if isinstance(value, Mapping):
        return tuple(
            parameter
            for item in value.values()
            for parameter in _collect_parameters(item)
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            parameter for item in value for parameter in _collect_parameters(item)
        )
    return ()


def _resolve(value: object, parameters: Mapping[str, object]):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


def _acceptance(efficiency, veto, data: dict[str, object]) -> jnp.ndarray:
    size = int(jnp.asarray(next(iter(data.values()))).shape[0])
    values = jnp.ones((size,), dtype=jnp.float64)
    if efficiency is not None:
        values = values * jnp.asarray(efficiency(data))
    if veto is not None:
        values = values * jnp.asarray(veto(data), dtype=values.dtype)
    return values


@dataclass(frozen=True)
class CPBackgroundSpec:
    """One CP background shape, normalized automatically over both charges."""

    name: str
    plus_shape: object
    minus_shape: object | None = None
    fraction: object | None = None
    yield_: object | None = None
    plus_normalization_sample: PhaseSpaceSample | None = None
    minus_normalization_sample: PhaseSpaceSample | None = None
    apply_veto: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CP background name must be non-empty")
        if not callable(self.plus_shape):
            raise TypeError("plus_shape must be callable on an event-data mapping")
        if self.minus_shape is not None and not callable(self.minus_shape):
            raise TypeError("minus_shape must be callable on an event-data mapping")
        if self.fraction is not None and self.yield_ is not None:
            raise ValueError("a CP background cannot define both fraction and yield")

    @property
    def resolved_minus_shape(self):
        return self.plus_shape if self.minus_shape is None else self.minus_shape


@dataclass(frozen=True)
class CPFitSession:
    """Compose a simultaneous direct-CP fit with minimal boilerplate.

    Signal efficiencies and vetoes are folded into both the event numerators and
    the deterministic normalization caches. Background specifications are
    evaluated and normalized automatically over the joint charge-Dalitz space.
    """

    plus_model: object
    minus_model: object
    plus_data: PhaseSpaceSample
    minus_data: PhaseSpaceSample
    plus_efficiency: object | None = None
    minus_efficiency: object | None = None
    plus_veto: object | None = None
    minus_veto: object | None = None
    backgrounds: tuple[CPBackgroundSpec | CPBackgroundCategory, ...] = ()
    signal_fraction: object | None = None
    extended: bool = False
    signal_yield: object | None = None
    constraints: tuple[object, ...] = ()

    @classmethod
    def from_root(
        cls,
        plus_model: object,
        minus_model: object,
        plus_file: str | Path,
        plus_tree: str,
        minus_file: str | Path,
        minus_tree: str,
        *,
        plus_root_kwargs: Mapping[str, object] | None = None,
        minus_root_kwargs: Mapping[str, object] | None = None,
        **session_kwargs,
    ) -> "CPFitSession":
        plus_data = read_phase_space_sample(
            plus_file, plus_tree, **({} if plus_root_kwargs is None else dict(plus_root_kwargs))
        )
        minus_data = read_phase_space_sample(
            minus_file, minus_tree, **({} if minus_root_kwargs is None else dict(minus_root_kwargs))
        )
        return cls(plus_model, minus_model, plus_data, minus_data, **session_kwargs)

    def with_efficiency(self, plus_efficiency: object | None, minus_efficiency: object | None = None) -> "CPFitSession":
        return replace(self, plus_efficiency=plus_efficiency, minus_efficiency=plus_efficiency if minus_efficiency is None else minus_efficiency)

    def with_veto(self, plus_veto: object | None, minus_veto: object | None = None) -> "CPFitSession":
        return replace(self, plus_veto=plus_veto, minus_veto=plus_veto if minus_veto is None else minus_veto)

    def with_background(self, name: str, plus_shape: object, *, minus_shape: object | None = None, fraction: object | None = None, yield_: object | None = None, plus_normalization_sample: PhaseSpaceSample | None = None, minus_normalization_sample: PhaseSpaceSample | None = None, apply_veto: bool = True) -> "CPFitSession":
        spec = CPBackgroundSpec(name=name, plus_shape=plus_shape, minus_shape=minus_shape, fraction=fraction, yield_=yield_, plus_normalization_sample=plus_normalization_sample, minus_normalization_sample=minus_normalization_sample, apply_veto=apply_veto)
        return replace(self, backgrounds=self.backgrounds + (spec,))

    def with_constraint(self, constraint: object) -> "CPFitSession":
        return replace(self, constraints=self.constraints + (constraint,))

    @cached_property
    def plus_acceptance_data(self) -> jnp.ndarray:
        return _acceptance(self.plus_efficiency, self.plus_veto, self.plus_data.as_dict())

    @cached_property
    def minus_acceptance_data(self) -> jnp.ndarray:
        return _acceptance(self.minus_efficiency, self.minus_veto, self.minus_data.as_dict())

    @cached_property
    def plus_acceptance_normalization(self) -> jnp.ndarray:
        sample = self.plus_model.normalization_sample
        return _acceptance(self.plus_efficiency, self.plus_veto, sample.as_dict())

    @cached_property
    def minus_acceptance_normalization(self) -> jnp.ndarray:
        sample = self.minus_model.normalization_sample
        return _acceptance(self.minus_efficiency, self.minus_veto, sample.as_dict())

    @cached_property
    def plus_cache(self):
        return self.plus_model.prepare_cache(self.plus_data, self.plus_model.normalization_sample, efficiency_normalization=self.plus_acceptance_normalization)

    @cached_property
    def minus_cache(self):
        return self.minus_model.prepare_cache(self.minus_data, self.minus_model.normalization_sample, efficiency_normalization=self.minus_acceptance_normalization)

    @staticmethod
    def _evaluate_shape(shape: object, data: dict) -> jnp.ndarray:
        values = jnp.asarray(shape(data))
        size = int(jnp.asarray(next(iter(data.values()))).shape[0])
        if values.shape != (size,):
            raise ValueError(f"background shape must return one value per event, got {values.shape} for {size} events")
        return values

    def _build_background(self, background: CPBackgroundSpec | CPBackgroundCategory) -> CPBackgroundCategory:
        if isinstance(background, CPBackgroundCategory):
            return background
        plus_norm = self.plus_model.normalization_sample if background.plus_normalization_sample is None else background.plus_normalization_sample
        minus_norm = self.minus_model.normalization_sample if background.minus_normalization_sample is None else background.minus_normalization_sample
        plus_shape = background.plus_shape
        minus_shape = background.resolved_minus_shape
        plus_data_dict = self.plus_data.as_dict(); minus_data_dict = self.minus_data.as_dict()
        plus_norm_dict = plus_norm.as_dict(); minus_norm_dict = minus_norm.as_dict()
        plus_values = self._evaluate_shape(plus_shape, plus_data_dict)
        minus_values = self._evaluate_shape(minus_shape, minus_data_dict)
        plus_norm_values = self._evaluate_shape(plus_shape, plus_norm_dict)
        minus_norm_values = self._evaluate_shape(minus_shape, minus_norm_dict)
        if background.apply_veto:
            if self.plus_veto is not None:
                plus_values = plus_values * jnp.asarray(self.plus_veto(plus_data_dict)); plus_norm_values = plus_norm_values * jnp.asarray(self.plus_veto(plus_norm_dict))
            if self.minus_veto is not None:
                minus_values = minus_values * jnp.asarray(self.minus_veto(minus_data_dict)); minus_norm_values = minus_norm_values * jnp.asarray(self.minus_veto(minus_norm_dict))
        plus_normalization = jnp.mean(jnp.asarray(plus_norm.weights) * plus_norm_values)
        minus_normalization = jnp.mean(jnp.asarray(minus_norm.weights) * minus_norm_values)
        return CPBackgroundCategory(name=background.name, plus_values=plus_values, minus_values=minus_values, plus_normalization=plus_normalization, minus_normalization=minus_normalization, fraction=background.fraction, yield_=background.yield_)

    @cached_property
    def background_categories(self) -> tuple[CPBackgroundCategory, ...]:
        return tuple(self._build_background(background) for background in self.backgrounds)

    @cached_property
    def base_objective(self) -> CPJointNLL:
        return CPJointNLL(self.plus_cache, self.minus_cache, plus_efficiency=self.plus_acceptance_data, minus_efficiency=self.minus_acceptance_data, background_categories=self.background_categories, signal_fraction=self.signal_fraction, extended=self.extended, signal_yield=self.signal_yield)

    @cached_property
    def objective(self):
        nll: object = self.base_objective
        if self.constraints:
            nll = ConstrainedNLL(nll, *self.constraints)
        return nll

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        candidates: list[Parameter] = list(getattr(self.plus_model, "parameters", ()))
        candidates.extend(getattr(self.minus_model, "parameters", ()))
        candidates.extend(_collect_parameters(self.signal_fraction)); candidates.extend(_collect_parameters(self.signal_yield)); candidates.extend(_collect_parameters(self.backgrounds)); candidates.extend(_collect_parameters(self.constraints))
        unique: dict[str, Parameter] = {}
        for parameter in candidates:
            previous = unique.get(parameter.name)
            if previous is not None and previous != parameter:
                raise ValueError(f"conflicting definitions for fit parameter {parameter.name!r}")
            unique[parameter.name] = parameter
        return tuple(unique.values())

    def minimizer(self, *, tolerance: float = 1e-4, verbose: int = 0) -> Minimizer:
        return Minimizer(self.objective, self.parameters, tolerance=tolerance, verbose=verbose)

    def fit(self, start_values: Mapping[str, float] | None = None, *, simplex: bool = False, ncall: int | None = None, tolerance: float = 1e-4, verbose: int = 0):
        return self.minimizer(tolerance=tolerance, verbose=verbose).fit(start_values=start_values, simplex=simplex, ncall=ncall)

    def fit_multistart(self, n_starts: int = 20, *, seed: int | None = None, include_default: bool = False, simplex: bool = False, tolerance: float = 1e-4, verbose: int = 0):
        return self.minimizer(tolerance=tolerance, verbose=verbose).fit_multistart(n_starts=n_starts, seed=seed, include_default=include_default, simplex=simplex)

    def result_values(self, result) -> dict[str, float]:
        values: dict[str, float] = {}
        for parameter in self.parameters:
            values[parameter.name] = float(parameter.value) if parameter.fixed else float(result.values[parameter.name])
        return values

    def print_result(self, result, *, precision: int = 6) -> dict[str, float]:
        values = self.result_values(result)
        print(f"valid={bool(result.valid)}  NLL={float(result.fval):.{precision}f}")
        print(f"{'parameter':24s} {'value':>16s} {'error':>16s}")
        for parameter in self.parameters:
            value = values[parameter.name]; error = 0.0 if parameter.fixed else float(result.errors[parameter.name])
            print(f"{parameter.name:24s} {value:16.{precision}g} {error:16.{precision}g}")
        pplus, pminus = self.base_objective.charge_probabilities(values)
        print(f"predicted charge fractions: B+={float(pplus):.6f}  B-={float(pminus):.6f}")
        return values

    def print_fit_fractions(self, result, *, acceptance_weighted: bool = False, include_interference: bool = False, precision: int = 3) -> dict[str, dict[str, float]]:
        values = self.result_values(result)
        plus_eff = self.plus_efficiency if acceptance_weighted else None; minus_eff = self.minus_efficiency if acceptance_weighted else None
        print("B+ fit fractions")
        plus = self.plus_model.print_fit_fractions(values, efficiency=plus_eff, include_interference=include_interference, precision=precision)
        print("\nB- fit fractions")
        minus = self.minus_model.print_fit_fractions(values, efficiency=minus_eff, include_interference=include_interference, precision=precision)
        return {"plus": plus, "minus": minus}

    def report(self, result, *, include_fit_fractions: bool = True, acceptance_weighted_fractions: bool = False, include_correlation: bool = True) -> dict[str, object]:
        values = self.print_result(result)
        errors = {p.name: (0.0 if p.fixed else float(result.errors[p.name])) for p in self.parameters}
        pplus, pminus = self.base_objective.charge_probabilities(values)
        report: dict[str, object] = {"valid": bool(result.valid), "nll": float(result.fval), "edm": float(result.fmin.edm), "nfcn": int(result.nfcn), "values": values, "errors": errors, "charge_probabilities": {"plus": float(pplus), "minus": float(pminus)}}
        if include_fit_fractions:
            report["fit_fractions"] = self.print_fit_fractions(result, acceptance_weighted=acceptance_weighted_fractions)
        if include_correlation and getattr(result, "covariance", None) is not None:
            correlation = result.covariance.correlation(); free = [p.name for p in self.parameters if not p.fixed]
            report["correlation"] = {first: {second: float(correlation[first, second]) for second in free} for first in free}
        return report

    def _projection_components(self, values: Mapping[str, float], charge: str) -> list[tuple[str, PhaseSpaceSample, np.ndarray]]:
        if charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")
        model = self.plus_model if charge == "plus" else self.minus_model
        norm = model.normalization_sample
        acceptance = self.plus_acceptance_normalization if charge == "plus" else self.minus_acceptance_normalization
        _, integral_plus = self.plus_cache.evaluate(values); _, integral_minus = self.minus_cache.evaluate(values)
        joint_signal_norm = integral_plus + integral_minus
        norm_intensity = jnp.asarray(model.intensity(norm.as_dict(), values)); signal_density = acceptance * norm_intensity / joint_signal_norm
        total_events = self.plus_data.size + self.minus_data.size
        if self.extended:
            signal_scale = float(_resolve(self.signal_yield, values))
        elif self.background_categories:
            signal_scale = total_events * float(_resolve(self.signal_fraction, values))
        else:
            signal_scale = float(total_events)
        components = [("signal", norm, np.asarray(signal_scale * norm.weights * signal_density / norm.size))]
        if not self.background_categories:
            return components
        if self.extended:
            bg_scales = [float(_resolve(category.yield_, values)) for category in self.background_categories]
        else:
            bg_total = total_events * (1.0 - float(_resolve(self.signal_fraction, values)))
            weights = np.asarray(self.base_objective.background_weights(values), dtype=float); bg_scales = [bg_total * float(weight) for weight in weights]
        for source, category, scale in zip(self.backgrounds, self.background_categories, bg_scales):
            if not isinstance(source, CPBackgroundSpec):
                continue
            shape = source.plus_shape if charge == "plus" else source.resolved_minus_shape
            bg_norm_sample = (source.plus_normalization_sample or self.plus_model.normalization_sample) if charge == "plus" else (source.minus_normalization_sample or self.minus_model.normalization_sample)
            raw = jnp.asarray(shape(bg_norm_sample.as_dict()))
            if source.apply_veto:
                veto = self.plus_veto if charge == "plus" else self.minus_veto
                if veto is not None:
                    raw = raw * jnp.asarray(veto(bg_norm_sample.as_dict()))
            density = raw / category.normalization
            components.append((category.name, bg_norm_sample, np.asarray(scale * bg_norm_sample.weights * density / bg_norm_sample.size)))
        return components

    def plot_projection(self, result, variable: str = "s13", *, bins: int = 60, range: tuple[float, float] | None = None, show_components: bool = True, log_scale: bool = False, axes=None):
        """Plot B+/B- data as black points with errors and fit as lines.

        ``log_scale=True`` enables a logarithmic y axis. Invariant-mass-squared
        projections are labelled as Candidates per actual uniform bin width.
        """
        import matplotlib.pyplot as plt
        values = self.result_values(result)
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        for ax, charge, data in zip(axes, ("plus", "minus"), (self.plus_data, self.minus_data)):
            data_values = np.asarray(getattr(data, variable))
            hist_range = range or (float(np.min(data_values)), float(np.max(data_values)))
            edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
            charge_label = f"B{'+' if charge == 'plus' else '-'} data"
            unit = r"GeV$^2$" if variable in ("s12", "s13", "s23") else ""
            plot_binned_data(data_values, bins=edges, ax=ax, label=charge_label, unit=unit, log_scale=log_scale)
            total = np.zeros(bins, dtype=float)
            for name, sample, weights in self._projection_components(values, charge):
                counts, _ = np.histogram(np.asarray(getattr(sample, variable)), bins=edges, weights=weights)
                total += counts
                if show_components:
                    ax.stairs(counts, edges, label=name)
            ax.stairs(total, edges, label="total fit", linewidth=2.0)
            ax.set_xlabel(rf"${variable}$ [GeV$^2$]")
            ax.legend()
        return axes


__all__ = ["CPBackgroundSpec", "CPFitSession"]
