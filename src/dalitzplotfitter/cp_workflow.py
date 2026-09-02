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
        return tuple(parameter for item in value.values() for parameter in _collect_parameters(item))
    if isinstance(value, (tuple, list)):
        return tuple(parameter for item in value for parameter in _collect_parameters(item))
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


def _joint_scaled_weights(plus_sample, plus_density, minus_sample, minus_density, scale: float):
    plus_raw = np.asarray(plus_sample.weights, dtype=float) * np.asarray(plus_density, dtype=float)
    minus_raw = np.asarray(minus_sample.weights, dtype=float) * np.asarray(minus_density, dtype=float)
    total = float(np.sum(plus_raw) + np.sum(minus_raw))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("joint projection density has non-positive or non-finite integral")
    factor = float(scale) / total
    return factor * plus_raw, factor * minus_raw


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
    def from_root(cls, plus_model, minus_model, plus_file, plus_tree, minus_file, minus_tree, *, plus_root_kwargs=None, minus_root_kwargs=None, **session_kwargs):
        plus_data = read_phase_space_sample(plus_file, plus_tree, **({} if plus_root_kwargs is None else dict(plus_root_kwargs)))
        minus_data = read_phase_space_sample(minus_file, minus_tree, **({} if minus_root_kwargs is None else dict(minus_root_kwargs)))
        return cls(plus_model, minus_model, plus_data, minus_data, **session_kwargs)

    def with_efficiency(self, plus_efficiency, minus_efficiency=None):
        return replace(self, plus_efficiency=plus_efficiency, minus_efficiency=plus_efficiency if minus_efficiency is None else minus_efficiency)

    def with_veto(self, plus_veto, minus_veto=None):
        return replace(self, plus_veto=plus_veto, minus_veto=plus_veto if minus_veto is None else minus_veto)

    def with_background(self, name, plus_shape, *, minus_shape=None, fraction=None, yield_=None, plus_normalization_sample=None, minus_normalization_sample=None, apply_veto=True):
        return replace(self, backgrounds=self.backgrounds + (CPBackgroundSpec(name, plus_shape, minus_shape, fraction, yield_, plus_normalization_sample, minus_normalization_sample, apply_veto),))

    def with_constraint(self, constraint):
        return replace(self, constraints=self.constraints + (constraint,))

    @cached_property
    def plus_acceptance_data(self):
        return _acceptance(self.plus_efficiency, self.plus_veto, self.plus_data.as_dict())

    @cached_property
    def minus_acceptance_data(self):
        return _acceptance(self.minus_efficiency, self.minus_veto, self.minus_data.as_dict())

    @cached_property
    def plus_acceptance_normalization(self):
        s = self.plus_model.normalization_sample
        return _acceptance(self.plus_efficiency, self.plus_veto, s.as_dict())

    @cached_property
    def minus_acceptance_normalization(self):
        s = self.minus_model.normalization_sample
        return _acceptance(self.minus_efficiency, self.minus_veto, s.as_dict())

    @cached_property
    def plus_cache(self):
        return self.plus_model.prepare_cache(self.plus_data, self.plus_model.normalization_sample, efficiency_normalization=self.plus_acceptance_normalization)

    @cached_property
    def minus_cache(self):
        return self.minus_model.prepare_cache(self.minus_data, self.minus_model.normalization_sample, efficiency_normalization=self.minus_acceptance_normalization)

    @staticmethod
    def _evaluate_shape(shape, data):
        values = jnp.asarray(shape(data))
        size = int(jnp.asarray(next(iter(data.values()))).shape[0])
        if values.shape != (size,):
            raise ValueError(f"background shape must return one value per event, got {values.shape} for {size} events")
        return values

    def _build_background(self, background):
        if isinstance(background, CPBackgroundCategory):
            return background
        plus_norm = self.plus_model.normalization_sample if background.plus_normalization_sample is None else background.plus_normalization_sample
        minus_norm = self.minus_model.normalization_sample if background.minus_normalization_sample is None else background.minus_normalization_sample
        plus_shape = background.plus_shape; minus_shape = background.resolved_minus_shape
        pd, md, pn, mn = self.plus_data.as_dict(), self.minus_data.as_dict(), plus_norm.as_dict(), minus_norm.as_dict()
        pv, mv = self._evaluate_shape(plus_shape, pd), self._evaluate_shape(minus_shape, md)
        pnv, mnv = self._evaluate_shape(plus_shape, pn), self._evaluate_shape(minus_shape, mn)
        if background.apply_veto:
            if self.plus_veto is not None:
                pv *= jnp.asarray(self.plus_veto(pd)); pnv *= jnp.asarray(self.plus_veto(pn))
            if self.minus_veto is not None:
                mv *= jnp.asarray(self.minus_veto(md)); mnv *= jnp.asarray(self.minus_veto(mn))
        return CPBackgroundCategory(background.name, pv, mv, jnp.mean(plus_norm.weights * pnv), jnp.mean(minus_norm.weights * mnv), background.fraction, background.yield_)

    @cached_property
    def background_categories(self):
        return tuple(self._build_background(b) for b in self.backgrounds)

    @cached_property
    def base_objective(self):
        return CPJointNLL(self.plus_cache, self.minus_cache, plus_efficiency=self.plus_acceptance_data, minus_efficiency=self.minus_acceptance_data, background_categories=self.background_categories, signal_fraction=self.signal_fraction, extended=self.extended, signal_yield=self.signal_yield)

    @cached_property
    def objective(self):
        return ConstrainedNLL(self.base_objective, *self.constraints) if self.constraints else self.base_objective

    @property
    def parameters(self):
        candidates = list(getattr(self.plus_model, "parameters", ())) + list(getattr(self.minus_model, "parameters", ()))
        candidates.extend(_collect_parameters(self.signal_fraction)); candidates.extend(_collect_parameters(self.signal_yield)); candidates.extend(_collect_parameters(self.backgrounds)); candidates.extend(_collect_parameters(self.constraints))
        unique = {}
        for p in candidates:
            if p.name in unique and unique[p.name] != p:
                raise ValueError(f"conflicting definitions for fit parameter {p.name!r}")
            unique[p.name] = p
        return tuple(unique.values())

    def minimizer(self, *, tolerance=1e-4, verbose=0):
        return Minimizer(self.objective, self.parameters, tolerance=tolerance, verbose=verbose)

    def fit(self, start_values=None, *, simplex=False, ncall=None, tolerance=1e-4, verbose=0):
        return self.minimizer(tolerance=tolerance, verbose=verbose).fit(start_values=start_values, simplex=simplex, ncall=ncall)

    def fit_multistart(self, n_starts=20, *, seed=None, include_default=False, simplex=False, tolerance=1e-4, verbose=0):
        return self.minimizer(tolerance=tolerance, verbose=verbose).fit_multistart(n_starts=n_starts, seed=seed, include_default=include_default, simplex=simplex)

    def result_values(self, result):
        return {p.name: (float(p.value) if p.fixed else float(result.values[p.name])) for p in self.parameters}

    def print_result(self, result, *, precision=6):
        values = self.result_values(result)
        print(f"valid={bool(result.valid)}  NLL={float(result.fval):.{precision}f}")
        print(f"{'parameter':24s} {'value':>16s} {'error':>16s}")
        for p in self.parameters:
            print(f"{p.name:24s} {values[p.name]:16.{precision}g} {(0.0 if p.fixed else float(result.errors[p.name])):16.{precision}g}")
        pplus, pminus = self.base_objective.charge_probabilities(values)
        print(f"predicted charge fractions: B+={float(pplus):.6f}  B-={float(pminus):.6f}")
        return values

    def print_fit_fractions(self, result, *, acceptance_weighted=False, include_interference=False, precision=3):
        values = self.result_values(result)
        print("B+ fit fractions")
        plus = self.plus_model.print_fit_fractions(values, efficiency=self.plus_efficiency if acceptance_weighted else None, include_interference=include_interference, precision=precision)
        print("\nB- fit fractions")
        minus = self.minus_model.print_fit_fractions(values, efficiency=self.minus_efficiency if acceptance_weighted else None, include_interference=include_interference, precision=precision)
        return {"plus": plus, "minus": minus}

    def report(self, result, *, include_fit_fractions=True, acceptance_weighted_fractions=False, include_correlation=True):
        values = self.print_result(result)
        errors = {p.name: (0.0 if p.fixed else float(result.errors[p.name])) for p in self.parameters}
        pplus, pminus = self.base_objective.charge_probabilities(values)
        out = {"valid": bool(result.valid), "nll": float(result.fval), "edm": float(result.fmin.edm), "nfcn": int(result.nfcn), "values": values, "errors": errors, "charge_probabilities": {"plus": float(pplus), "minus": float(pminus)}}
        if include_fit_fractions:
            out["fit_fractions"] = self.print_fit_fractions(result, acceptance_weighted=acceptance_weighted_fractions)
        if include_correlation and getattr(result, "covariance", None) is not None:
            c = result.covariance.correlation(); free = [p.name for p in self.parameters if not p.fixed]
            out["correlation"] = {a: {b: float(c[a,b]) for b in free} for a in free}
        return out

    def _projection_components_pair(self, values, plus_sample, minus_sample):
        plus_acc = _acceptance(self.plus_efficiency, self.plus_veto, plus_sample.as_dict())
        minus_acc = _acceptance(self.minus_efficiency, self.minus_veto, minus_sample.as_dict())
        _, integral_plus = self.plus_cache.evaluate(values); _, integral_minus = self.minus_cache.evaluate(values)
        joint_signal_norm = integral_plus + integral_minus
        plus_density = plus_acc * self.plus_model.intensity(plus_sample.as_dict(), values) / joint_signal_norm
        minus_density = minus_acc * self.minus_model.intensity(minus_sample.as_dict(), values) / joint_signal_norm
        total_events = self.plus_data.size + self.minus_data.size
        signal_scale = float(_resolve(self.signal_yield, values)) if self.extended else (total_events * float(_resolve(self.signal_fraction, values)) if self.background_categories else float(total_events))
        plus_w, minus_w = _joint_scaled_weights(plus_sample, plus_density, minus_sample, minus_density, signal_scale)
        plus_components = [("signal", plus_sample, plus_w)]; minus_components = [("signal", minus_sample, minus_w)]
        if not self.background_categories:
            return plus_components, minus_components
        if self.extended:
            bg_scales = [float(_resolve(c.yield_, values)) for c in self.background_categories]
        else:
            bg_total = total_events * (1.0 - float(_resolve(self.signal_fraction, values)))
            bw = np.asarray(self.base_objective.background_weights(values), dtype=float)
            bg_scales = [bg_total * float(w) for w in bw]
        for source, category, scale in zip(self.backgrounds, self.background_categories, bg_scales):
            if not isinstance(source, CPBackgroundSpec):
                continue
            pr = jnp.asarray(source.plus_shape(plus_sample.as_dict())); mr = jnp.asarray(source.resolved_minus_shape(minus_sample.as_dict()))
            if source.apply_veto:
                if self.plus_veto is not None: pr *= jnp.asarray(self.plus_veto(plus_sample.as_dict()))
                if self.minus_veto is not None: mr *= jnp.asarray(self.minus_veto(minus_sample.as_dict()))
            pw, mw = _joint_scaled_weights(plus_sample, pr / category.normalization, minus_sample, mr / category.normalization, scale)
            plus_components.append((category.name, plus_sample, pw)); minus_components.append((category.name, minus_sample, mw))
        return plus_components, minus_components

    def plot_projection(self, result, variable="s13", *, bins=60, range=None, show_components=True, log_scale=False, projection_size=250_000, projection_seed=20260901, axes=None):
        """Plot smooth B+/B- projections without histogramming quadrature nodes.

        Two weighted phase-space MC samples are used only for rendering. Their
        component weights are normalized jointly across charges, preserving the
        integrated charge asymmetry of the fitted model.
        """
        import matplotlib.pyplot as plt
        values = self.result_values(result)
        if axes is None:
            _, axes = plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
        plus_sample = self.plus_model.generate_phase_space(projection_size, seed=projection_seed)
        minus_sample = self.minus_model.generate_phase_space(projection_size, seed=projection_seed + 1)
        plus_components, minus_components = self._projection_components_pair(values, plus_sample, minus_sample)
        for ax, charge, data, components in zip(axes, ("plus","minus"), (self.plus_data,self.minus_data), (plus_components,minus_components)):
            dv = np.asarray(getattr(data, variable)); hist_range = range or (float(np.min(dv)), float(np.max(dv))); edges = np.linspace(hist_range[0], hist_range[1], bins+1)
            unit = r"GeV$^2$" if variable in ("s12","s13","s23") else ""
            plot_binned_data(dv, bins=edges, ax=ax, label=f"B{'+' if charge=='plus' else '-'} data", unit=unit, log_scale=log_scale)
            total = np.zeros(bins)
            for name, sample, weights in components:
                counts, _ = np.histogram(np.asarray(getattr(sample, variable)), bins=edges, weights=weights); total += counts
                if show_components: ax.stairs(counts, edges, label=name)
            ax.stairs(total, edges, label="total fit", linewidth=2.0); ax.set_xlabel(rf"${variable}$ [GeV$^2$]"); ax.legend()
        return axes


__all__ = ["CPBackgroundSpec", "CPFitSession"]
