"""User-facing high-level workflow helpers.

The low-level PDF, likelihood, cache and minimizer classes remain available.
This module only composes them for common analysis workflows with less boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.background import BackgroundCategory
from dalitzplotfitter.constraints import ConstrainedNLL
from dalitzplotfitter.efficiency import UnityEfficiency
from dalitzplotfitter.fit import Minimizer, Parameter
from dalitzplotfitter.integration import GridIntegrator
from dalitzplotfitter.io import read_phase_space_sample
from dalitzplotfitter.kinematics import PhaseSpaceSample
from dalitzplotfitter.likelihood import MultiBackgroundNLL, UnbinnedNLL
from dalitzplotfitter.pdf import SignalPDF
from dalitzplotfitter.plotting import plot_binned_data


@dataclass(frozen=True)
class BackgroundSpec:
    """Background shape that is normalized automatically on the fit measure."""

    name: str
    shape: object
    fraction: object | None = None
    yield_: object | None = None
    normalization_sample: PhaseSpaceSample | None = None
    apply_veto: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("background name must be non-empty")
        if not callable(self.shape):
            raise TypeError("BackgroundSpec shape must be callable on an event-data mapping")
        if self.fraction is not None and self.yield_ is not None:
            raise ValueError("a background cannot define both fraction and yield")


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


@dataclass(frozen=True)
class FitSession:
    """Compose a common single-sample amplitude fit in a few lines."""

    model: object
    data: PhaseSpaceSample
    efficiency: object | None = None
    veto: object | None = None
    backgrounds: tuple[BackgroundSpec | BackgroundCategory, ...] = ()
    signal_fraction: object | None = None
    extended: bool = False
    signal_yield: object | None = None
    constraints: tuple[object, ...] = ()

    @classmethod
    def from_root(
        cls,
        model: object,
        file_path: str | Path,
        tree: str,
        *,
        efficiency: object | None = None,
        veto: object | None = None,
        backgrounds: Sequence[BackgroundSpec | BackgroundCategory] = (),
        signal_fraction: object | None = None,
        extended: bool = False,
        signal_yield: object | None = None,
        constraints: Sequence[object] = (),
        **root_kwargs,
    ) -> "FitSession":
        data = read_phase_space_sample(file_path, tree, **root_kwargs)
        return cls(
            model=model,
            data=data,
            efficiency=efficiency,
            veto=veto,
            backgrounds=tuple(backgrounds),
            signal_fraction=signal_fraction,
            extended=extended,
            signal_yield=signal_yield,
            constraints=tuple(constraints),
        )

    def with_efficiency(self, efficiency: object | None) -> "FitSession":
        return replace(self, efficiency=efficiency)

    def with_veto(self, veto: object | None) -> "FitSession":
        return replace(self, veto=veto)

    def with_background(
        self,
        name: str,
        shape: object,
        *,
        fraction: object | None = None,
        yield_: object | None = None,
        normalization_sample: PhaseSpaceSample | None = None,
        apply_veto: bool = True,
    ) -> "FitSession":
        spec = BackgroundSpec(
            name=name,
            shape=shape,
            fraction=fraction,
            yield_=yield_,
            normalization_sample=normalization_sample,
            apply_veto=apply_veto,
        )
        return replace(self, backgrounds=self.backgrounds + (spec,))

    def with_constraint(self, constraint: object) -> "FitSession":
        return replace(self, constraints=self.constraints + (constraint,))

    @cached_property
    def signal_pdf(self) -> SignalPDF:
        sample = self.model.normalization_sample

        def intensity(data, parameters):
            return self.model.intensity(data, parameters)

        return SignalPDF(
            intensity=intensity,
            integrator=GridIntegrator(sample),
            efficiency=UnityEfficiency() if self.efficiency is None else self.efficiency,
            veto=self.veto,
        )

    @staticmethod
    def _evaluate_shape(shape: object, data: dict) -> jnp.ndarray:
        values = jnp.asarray(shape(data))
        size = int(jnp.asarray(next(iter(data.values()))).shape[0])
        if values.shape != (size,):
            raise ValueError(
                f"background shape must return one value per event, got {values.shape} for {size} events"
            )
        return values

    def _build_background(self, background: BackgroundSpec | BackgroundCategory) -> BackgroundCategory:
        if isinstance(background, BackgroundCategory):
            return background
        data_dict = self.data.as_dict()
        norm_sample = self.model.normalization_sample if background.normalization_sample is None else background.normalization_sample
        norm_dict = norm_sample.as_dict()
        data_values = self._evaluate_shape(background.shape, data_dict)
        norm_values = self._evaluate_shape(background.shape, norm_dict)
        if self.veto is not None and background.apply_veto:
            data_values = data_values * jnp.asarray(self.veto(data_dict), dtype=data_values.dtype)
            norm_values = norm_values * jnp.asarray(self.veto(norm_dict), dtype=norm_values.dtype)
        normalization = jnp.mean(jnp.asarray(norm_sample.weights) * norm_values)
        return BackgroundCategory(
            name=background.name,
            values=data_values,
            normalization=normalization,
            fraction=background.fraction,
            yield_=background.yield_,
        )

    @cached_property
    def background_categories(self) -> tuple[BackgroundCategory, ...]:
        return tuple(self._build_background(background) for background in self.backgrounds)

    @cached_property
    def base_objective(self):
        data = self.data.as_dict()
        if not self.background_categories and not self.extended:
            return UnbinnedNLL(self.signal_pdf.logpdf, data)
        return MultiBackgroundNLL(
            signal_density=lambda parameters: self.signal_pdf(data, parameters),
            backgrounds=self.background_categories,
            signal_fraction=self.signal_fraction,
            extended=self.extended,
            signal_yield=self.signal_yield,
        )

    @cached_property
    def objective(self):
        nll: object = self.base_objective
        if self.constraints:
            nll = ConstrainedNLL(nll, *self.constraints)
        return nll

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        candidates: list[Parameter] = list(getattr(self.model, "parameters", ()))
        candidates.extend(_collect_parameters(self.signal_fraction))
        candidates.extend(_collect_parameters(self.signal_yield))
        candidates.extend(_collect_parameters(self.backgrounds))
        candidates.extend(_collect_parameters(self.constraints))
        unique: dict[str, Parameter] = {}
        for parameter in candidates:
            previous = unique.get(parameter.name)
            if previous is not None and previous != parameter:
                raise ValueError(f"conflicting definitions for fit parameter {parameter.name!r}")
            unique[parameter.name] = parameter
        return tuple(unique.values())

    def minimizer(self, *, tolerance: float = 1e-4, verbose: int = 0) -> Minimizer:
        return Minimizer(self.objective, self.parameters, tolerance=tolerance, verbose=verbose)

    def fit(
        self,
        start_values: Mapping[str, float] | None = None,
        *,
        simplex: bool = False,
        ncall: int | None = None,
        tolerance: float = 1e-4,
        verbose: int = 0,
    ):
        return self.minimizer(tolerance=tolerance, verbose=verbose).fit(
            start_values=start_values,
            simplex=simplex,
            ncall=ncall,
        )

    def fit_multistart(
        self,
        n_starts: int = 20,
        *,
        seed: int | None = None,
        include_default: bool = False,
        simplex: bool = False,
        tolerance: float = 1e-4,
        verbose: int = 0,
    ):
        return self.minimizer(tolerance=tolerance, verbose=verbose).fit_multistart(
            n_starts=n_starts,
            seed=seed,
            include_default=include_default,
            simplex=simplex,
        )

    def result_values(self, result) -> dict[str, float]:
        values: dict[str, float] = {}
        for parameter in self.parameters:
            values[parameter.name] = float(parameter.value) if parameter.fixed else float(result.values[parameter.name])
        return values

    def print_result(self, result, *, precision: int = 6) -> dict[str, float]:
        if precision < 0:
            raise ValueError("precision must be non-negative")
        values = self.result_values(result)
        print(f"valid={bool(result.valid)}  NLL={float(result.fval):.{precision}f}")
        print(f"{'parameter':24s} {'value':>16s} {'error':>16s}")
        for parameter in self.parameters:
            value = values[parameter.name]
            error = 0.0 if parameter.fixed else float(result.errors[parameter.name])
            print(f"{parameter.name:24s} {value:16.{precision}g} {error:16.{precision}g}")
        return values

    def print_fit_fractions(
        self,
        result,
        *,
        acceptance_weighted: bool = False,
        include_interference: bool = False,
        precision: int = 3,
    ):
        return self.model.print_fit_fractions(
            self.result_values(result),
            efficiency=self.efficiency if acceptance_weighted else None,
            include_interference=include_interference,
            precision=precision,
        )

    def report(
        self,
        result,
        *,
        include_fit_fractions: bool = True,
        acceptance_weighted_fractions: bool = False,
        include_correlation: bool = True,
    ) -> dict[str, object]:
        """Print and return a compact fit summary."""

        values = self.print_result(result)
        errors = {
            p.name: (0.0 if p.fixed else float(result.errors[p.name])) for p in self.parameters
        }
        report: dict[str, object] = {
            "valid": bool(result.valid),
            "nll": float(result.fval),
            "edm": float(result.fmin.edm),
            "nfcn": int(result.nfcn),
            "values": values,
            "errors": errors,
        }
        if include_fit_fractions:
            report["fit_fractions"] = self.print_fit_fractions(
                result,
                acceptance_weighted=acceptance_weighted_fractions,
            )
        if include_correlation and getattr(result, "covariance", None) is not None:
            correlation = result.covariance.correlation()
            free = [p.name for p in self.parameters if not p.fixed]
            report["correlation"] = {
                first: {second: float(correlation[first, second]) for second in free}
                for first in free
            }
        return report

    def _projection_components(
        self, values: Mapping[str, float]
    ) -> list[tuple[str, PhaseSpaceSample, np.ndarray]]:
        norm = self.model.normalization_sample
        signal_density = jnp.asarray(self.signal_pdf(norm.as_dict(), values))
        if self.extended:
            signal_scale = float(_resolve(self.signal_yield, values))
        elif self.background_categories:
            signal_scale = self.data.size * float(_resolve(self.signal_fraction, values))
        else:
            signal_scale = float(self.data.size)
        components = [
            (
                "signal",
                norm,
                np.asarray(signal_scale * norm.weights * signal_density / norm.size),
            )
        ]
        if not self.background_categories:
            return components

        if self.extended:
            bg_scales = [float(_resolve(category.yield_, values)) for category in self.background_categories]
        else:
            bg_total = self.data.size * (1.0 - float(_resolve(self.signal_fraction, values)))
            weights = np.asarray(self.base_objective.background_weights(values), dtype=float)
            bg_scales = [bg_total * float(weight) for weight in weights]

        for source, category, scale in zip(self.backgrounds, self.background_categories, bg_scales):
            if not isinstance(source, BackgroundSpec):
                continue
            bg_norm_sample = source.normalization_sample or self.model.normalization_sample
            raw = jnp.asarray(source.shape(bg_norm_sample.as_dict()))
            if self.veto is not None and source.apply_veto:
                raw = raw * jnp.asarray(self.veto(bg_norm_sample.as_dict()))
            density = raw / category.normalization
            components.append(
                (
                    category.name,
                    bg_norm_sample,
                    np.asarray(scale * bg_norm_sample.weights * density / bg_norm_sample.size),
                )
            )
        return components

    def plot_projection(
        self,
        result,
        variable: str = "s13",
        *,
        bins: int = 60,
        range: tuple[float, float] | None = None,
        show_components: bool = True,
        ax=None,
    ):
        """Plot data as black points with errors and fitted model as lines."""

        import matplotlib.pyplot as plt

        values = self.result_values(result)
        data_values = np.asarray(getattr(self.data, variable))
        hist_range = range or (float(np.min(data_values)), float(np.max(data_values)))
        edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        plot_binned_data(data_values, bins=edges, ax=ax, label="data")
        total = np.zeros(bins, dtype=float)
        for name, sample, weights in self._projection_components(values):
            counts, _ = np.histogram(np.asarray(getattr(sample, variable)), bins=edges, weights=weights)
            total += counts
            if show_components:
                ax.stairs(counts, edges, label=name)
        ax.stairs(total, edges, label="total fit", linewidth=2.0)
        ax.set_xlabel(rf"${variable}$ [GeV$^2$]")
        ax.set_ylabel("events / bin")
        ax.legend()
        return ax


__all__ = ["BackgroundSpec", "FitSession"]
