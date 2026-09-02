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

from dalitzplotfitter.amplitude import PreparedAmplitudeCache
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
            raise ValueError("BackgroundSpec name must be non-empty")
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


def _acceptance(efficiency, veto, data: dict[str, object]) -> jnp.ndarray:
    """Evaluate the parameter-independent event acceptance once."""

    size = int(jnp.asarray(next(iter(data.values()))).shape[0])
    values = jnp.ones((size,), dtype=jnp.float64)
    if efficiency is not None:
        values = values * jnp.asarray(efficiency(data))
    if veto is not None:
        values = values * jnp.asarray(veto(data), dtype=values.dtype)
    return values


def _scaled_projection_weights(
    sample: PhaseSpaceSample,
    density,
    scale: float,
) -> np.ndarray:
    """Normalize MC projection weights to the requested component yield."""

    raw = np.asarray(sample.weights, dtype=float) * np.asarray(density, dtype=float)
    total = float(np.sum(raw))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("projection density has non-positive or non-finite integral")
    return float(scale) * raw / total


@dataclass(frozen=True)
class FitSession:
    """Compose a common single-sample amplitude fit in a few lines.

    The likelihood path uses :class:`PreparedAmplitudeCache`: fixed component
    dynamics and the fixed normalization matrix are prepared once, while only
    components owning floating dynamical parameters are reevaluated during the
    fit. Efficiency and veto values are likewise cached on the data and
    normalization samples.
    """

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
    def acceptance_data(self) -> jnp.ndarray:
        return _acceptance(self.efficiency, self.veto, self.data.as_dict())

    @cached_property
    def acceptance_normalization(self) -> jnp.ndarray:
        sample = self.model.normalization_sample
        return _acceptance(self.efficiency, self.veto, sample.as_dict())

    @cached_property
    def signal_cache(self):
        """Prepared amplitude cache used by every likelihood evaluation."""

        return self.model.prepare_cache(
            self.data,
            self.model.normalization_sample,
            efficiency_normalization=self.acceptance_normalization,
        )

    @cached_property
    def signal_pdf(self) -> SignalPDF:
        """Generic PDF retained for diagnostics and arbitrary projection samples.

        The fit objective itself does not use this slower generic path; it uses
        ``signal_cache`` so repeated likelihood evaluations do not recompute
        fixed dynamics or the full normalization integral.
        """

        sample = self.model.normalization_sample

        def intensity(data, parameters):
            return self.model.intensity(data, parameters)

        return SignalPDF(
            intensity=intensity,
            integrator=GridIntegrator(sample),
            efficiency=UnityEfficiency() if self.efficiency is None else self.efficiency,
            veto=self.veto,
        )

    @cached_property
    def _projection_samples(self) -> dict[tuple[int, int | None], PhaseSpaceSample]:
        """Phase-space samples reused by repeated projection calls."""

        return {}

    @cached_property
    def _projection_prepared(self) -> dict[int, tuple[PhaseSpaceSample, object | None, jnp.ndarray]]:
        """Prepared amplitudes/acceptance keyed by live projection-sample identity."""

        return {}

    def _get_projection_sample(
        self,
        size: int,
        seed: int | None,
    ) -> PhaseSpaceSample:
        size = int(size)
        if size < 1:
            raise ValueError("projection_size must be positive")
        resolved_seed = None if seed is None else int(seed)
        key = (size, resolved_seed)
        sample = self._projection_samples.get(key)
        if sample is None:
            sample = self.model.generate_phase_space(size, seed=resolved_seed)
            self._projection_samples[key] = sample
        return sample

    def _prepare_projection_sample(
        self,
        sample: PhaseSpaceSample,
    ) -> tuple[object | None, jnp.ndarray]:
        """Prepare projection amplitudes once and reuse the fit normalization.

        Coefficient-only fits already have the exact fixed normalization matrix
        and component scales in ``signal_cache``.  Projection rendering only
        needs component values on a new phase-space sample, so rebuilding the
        normalization integral (the old ``signal_pdf`` path) is unnecessary.
        """

        key = id(sample)
        prepared = self._projection_prepared.get(key)
        if prepared is not None and prepared[0] is sample:
            return prepared[1], prepared[2]

        acceptance = _acceptance(self.efficiency, self.veto, sample.as_dict())
        template = self.signal_cache
        projection_cache = None
        if template.is_compact:
            if template.component_scales is None:
                raise RuntimeError("compact signal cache is missing component scales")
            compact_data_kernel = None
            kernel_builder = getattr(self.model, "_compact_data_kernel", None)
            if callable(kernel_builder):
                compact_data_kernel = kernel_builder(
                    normalize_components=template.normalize_components
                )
            projection_cache = PreparedAmplitudeCache.prepare_from_fixed_normalization(
                template.components,
                data=sample.as_dict(),
                normalization_weights=template.normalization_weights,
                parameters=template.parameters,
                normalization_matrix_fixed=template.normalization_matrix_fixed,
                component_scales=template.component_scales,
                normalize_components=template.normalize_components,
                compact_data_kernel=compact_data_kernel,
            )

        self._projection_prepared[key] = (sample, projection_cache, acceptance)
        return projection_cache, acceptance

    def _projection_signal_density(
        self,
        sample: PhaseSpaceSample,
        values: Mapping[str, object],
    ) -> jnp.ndarray:
        projection_cache, acceptance = self._prepare_projection_sample(sample)
        if projection_cache is not None:
            intensity, normalization = projection_cache.evaluate(values)
            return acceptance * intensity / normalization

        # Floating-dynamics caches cannot yet be cloned data-only. Keep the
        # generic path for correctness until partial dynamic preparation exists.
        return jnp.asarray(self.signal_pdf(sample.as_dict(), values))

    def _cached_signal_density(self, parameters: Mapping[str, object]) -> jnp.ndarray:
        intensity, normalization = self.signal_cache.evaluate(parameters)
        return self.acceptance_data * intensity / normalization

    def _cached_signal_logpdf(
        self,
        data: dict[str, object],
        parameters: Mapping[str, object],
    ) -> jnp.ndarray:
        del data
        intensity, normalization = self.signal_cache.evaluate(parameters)
        numerator = self.acceptance_data * intensity
        return jnp.log(jnp.clip(numerator, min=1e-300)) - jnp.log(normalization)

    @staticmethod
    def _evaluate_shape(shape: object, data: dict) -> jnp.ndarray:
        values = jnp.asarray(shape(data))
        size = int(jnp.asarray(next(iter(data.values()))).shape[0])
        if values.shape != (size,):
            raise ValueError(
                f"background shape must return one value per event, got {values.shape} "
                f"for {size} events"
            )
        return values

    def _build_background(
        self,
        background: BackgroundSpec | BackgroundCategory,
    ) -> BackgroundCategory:
        if isinstance(background, BackgroundCategory):
            return background
        data_dict = self.data.as_dict()
        norm_sample = (
            self.model.normalization_sample
            if background.normalization_sample is None
            else background.normalization_sample
        )
        norm_dict = norm_sample.as_dict()
        data_values = self._evaluate_shape(background.shape, data_dict)
        norm_values = self._evaluate_shape(background.shape, norm_dict)
        if self.veto is not None and background.apply_veto:
            data_values = data_values * jnp.asarray(
                self.veto(data_dict), dtype=data_values.dtype
            )
            norm_values = norm_values * jnp.asarray(
                self.veto(norm_dict), dtype=norm_values.dtype
            )
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
        return tuple(
            self._build_background(background) for background in self.backgrounds
        )

    @cached_property
    def base_objective(self):
        # Materialize all parameter-independent signal arrays before the
        # minimizer JIT traces the objective. Creating a cached_property while
        # tracing would otherwise store JAX tracers in the session and makes
        # host-side cache validation illegal inside the traced function.
        _ = self.signal_cache
        _ = self.acceptance_data

        data = self.data.as_dict()
        if not self.background_categories and not self.extended:
            return UnbinnedNLL(self._cached_signal_logpdf, data)
        return MultiBackgroundNLL(
            signal_density=self._cached_signal_density,
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
                raise ValueError(
                    f"conflicting definitions for fit parameter {parameter.name!r}"
                )
            unique[parameter.name] = parameter
        return tuple(unique.values())

    def minimizer(
        self,
        *,
        tolerance: float = 1e-4,
        verbose: int = 0,
    ) -> Minimizer:
        return Minimizer(
            self.objective,
            self.parameters,
            tolerance=tolerance,
            verbose=verbose,
        )

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
        return {
            parameter.name: (
                float(parameter.value)
                if parameter.fixed
                else float(result.values[parameter.name])
            )
            for parameter in self.parameters
        }

    def print_result(self, result, *, precision: int = 6) -> dict[str, float]:
        if precision < 0:
            raise ValueError("precision must be non-negative")
        values = self.result_values(result)
        print(f"valid={bool(result.valid)}  NLL={float(result.fval):.{precision}f}")
        print(f"{'parameter':24s} {'value':>16s} {'error':>16s}")
        for parameter in self.parameters:
            value = values[parameter.name]
            error = 0.0 if parameter.fixed else float(result.errors[parameter.name])
            print(
                f"{parameter.name:24s} {value:16.{precision}g} "
                f"{error:16.{precision}g}"
            )
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
        values = self.print_result(result)
        errors = {
            parameter.name: (
                0.0
                if parameter.fixed
                else float(result.errors[parameter.name])
            )
            for parameter in self.parameters
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
            free = [parameter.name for parameter in self.parameters if not parameter.fixed]
            report["correlation"] = {
                first: {
                    second: float(correlation[first, second]) for second in free
                }
                for first in free
            }
        return report

    def _projection_components(
        self,
        values: Mapping[str, float],
        projection_sample: PhaseSpaceSample | None = None,
    ) -> list[tuple[str, PhaseSpaceSample, np.ndarray]]:
        sample = (
            self.model.normalization_sample
            if projection_sample is None
            else projection_sample
        )
        signal_density = self._projection_signal_density(sample, values)
        if self.extended:
            signal_scale = float(_resolve(self.signal_yield, values))
        elif self.background_categories:
            signal_scale = self.data.size * float(
                _resolve(self.signal_fraction, values)
            )
        else:
            signal_scale = float(self.data.size)
        components = [
            (
                "signal",
                sample,
                _scaled_projection_weights(sample, signal_density, signal_scale),
            )
        ]
        if not self.background_categories:
            return components
        if self.extended:
            bg_scales = [
                float(_resolve(category.yield_, values))
                for category in self.background_categories
            ]
        else:
            bg_total = self.data.size * (
                1.0 - float(_resolve(self.signal_fraction, values))
            )
            weights = np.asarray(
                self.base_objective.background_weights(values), dtype=float
            )
            bg_scales = [bg_total * float(weight) for weight in weights]
        for source, category, scale in zip(
            self.backgrounds,
            self.background_categories,
            bg_scales,
        ):
            if not isinstance(source, BackgroundSpec):
                continue
            bg_sample = (
                source.normalization_sample or self.model.normalization_sample
            ) if projection_sample is None else projection_sample
            raw = jnp.asarray(source.shape(bg_sample.as_dict()))
            if self.veto is not None and source.apply_veto:
                raw = raw * jnp.asarray(self.veto(bg_sample.as_dict()))
            density = raw / category.normalization
            components.append(
                (
                    category.name,
                    bg_sample,
                    _scaled_projection_weights(bg_sample, density, scale),
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
        log_scale: bool = False,
        projection_size: int = 100_000,
        projection_seed: int = 20260901,
        projection_sample: PhaseSpaceSample | None = None,
        ax=None,
    ):
        """Plot data and a smooth fitted projection.

        Fit/PDF normalization remains deterministic. A weighted phase-space MC
        sample is used only to render the one-dimensional model projection. The
        generated sample and coefficient-only prepared amplitudes are cached by
        the session, so plotting another invariant or changing bins does not
        repeat phase-space generation or fixed resonance dynamics.
        """

        import matplotlib.pyplot as plt

        values = self.result_values(result)
        data_values = np.asarray(getattr(self.data, variable))
        hist_range = range or (
            float(np.min(data_values)),
            float(np.max(data_values)),
        )
        edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        unit = r"GeV$^2$" if variable in ("s12", "s13", "s23") else ""
        plot_binned_data(
            data_values,
            bins=edges,
            ax=ax,
            label="data",
            unit=unit,
            log_scale=log_scale,
        )
        sample = (
            self._get_projection_sample(projection_size, projection_seed)
            if projection_sample is None
            else projection_sample
        )
        total = np.zeros(bins, dtype=float)
        for name, component_sample, weights in self._projection_components(
            values,
            sample,
        ):
            counts, _ = np.histogram(
                np.asarray(getattr(component_sample, variable)),
                bins=edges,
                weights=weights,
            )
            total += counts
            if show_components:
                ax.stairs(counts, edges, label=name)
        ax.stairs(total, edges, label="total fit", linewidth=2.0)
        ax.set_xlabel(rf"${variable}$ [GeV$^2$]")
        ax.legend()
        return ax


__all__ = ["BackgroundSpec", "FitSession"]
