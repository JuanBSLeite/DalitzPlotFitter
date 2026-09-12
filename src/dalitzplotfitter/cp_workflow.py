"""High-level workflow for simultaneous direct-CP Dalitz fits."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.background import CPBackgroundCategory
from dalitzplotfitter.constraints import ConstrainedNLL
from dalitzplotfitter.fit import Minimizer, Parameter
from dalitzplotfitter.goodness_of_fit import (
    BinnedChi2Result,
    PointToPointResult,
    chi2_from_histograms,
)
from dalitzplotfitter.goodness_of_fit import (
    point_to_point_dissimilarity as _point_to_point_dissimilarity,
)
from dalitzplotfitter.io import read_phase_space_sample
from dalitzplotfitter.kinematics import (
    PhaseSpaceSample,
    fold_thetaprime,
    invariants_to_square_dalitz,
)
from dalitzplotfitter.likelihood import CPJointNLL
from dalitzplotfitter.likelihood.cp import _signal_yield_pair
from dalitzplotfitter.observables.errors import _covariance_matrix
from dalitzplotfitter.plotting import _draw_pulls_1d, plot_binned_data
from dalitzplotfitter.sampling import weighted_resample


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
    for label, function in (("efficiency", efficiency), ("veto", veto)):
        if function is not None:
            array = jnp.asarray(function(data))
            if array.ndim == 0:
                array = jnp.full((size,), array)
            if array.shape != (size,):
                raise ValueError(f"CP {label} must have shape ({size},)")
            if bool(jnp.any(~jnp.isfinite(array) | (array < 0))):
                raise ValueError(f"CP {label} must be finite and non-negative")
            values = values * array
    return values


def _joint_scaled_weights(plus_sample, plus_density, minus_sample, minus_density, scale: float):
    if scale == 0:
        return np.zeros(plus_sample.size), np.zeros(minus_sample.size)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("projection yield must be finite and non-negative")
    if plus_sample.size == 0 or minus_sample.size == 0:
        raise ValueError("projection samples must be non-empty")
    plus_raw = np.asarray(plus_sample.weights, dtype=float) * np.asarray(plus_density, dtype=float) / plus_sample.size
    minus_raw = np.asarray(minus_sample.weights, dtype=float) * np.asarray(minus_density, dtype=float) / minus_sample.size
    total = float(np.sum(plus_raw) + np.sum(minus_raw))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("joint projection density has non-positive or non-finite integral")
    factor = float(scale) / total
    return factor * plus_raw, factor * minus_raw


@dataclass(frozen=True)
class CPBackgroundSpec:
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
        return self.plus_model.prepare_cache(self.plus_data, self.plus_model.normalization_sample, efficiency_normalization=(None if self.plus_efficiency is None and self.plus_veto is None else self.plus_acceptance_normalization))

    @cached_property
    def minus_cache(self):
        return self.minus_model.prepare_cache(self.minus_data, self.minus_model.normalization_sample, efficiency_normalization=(None if self.minus_efficiency is None and self.minus_veto is None else self.minus_acceptance_normalization))

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
        plus_shape = background.plus_shape
        minus_shape = background.resolved_minus_shape
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

    def minimizer(self, *, tolerance=1e-4, verbose=0, hessian="numerical"):
        return Minimizer(
            self.objective, self.parameters,
            tolerance=tolerance, verbose=verbose, hessian=hessian,
        )

    def fit(
        self, start_values=None, *, simplex=False, ncall=None, strategy=2,
        hesse=True, tolerance=1e-4, verbose=0, hessian="numerical",
    ):
        return self.minimizer(
            tolerance=tolerance, verbose=verbose, hessian=hessian,
        ).fit(
            start_values=start_values,
            simplex=simplex,
            ncall=ncall,
            strategy=strategy,
            hesse=hesse,
        )

    def fit_multistart(
        self, n_starts=20, *, seed=None, include_default=False, simplex=False,
        strategy=1, tolerance=1e-4, verbose=0, hessian="numerical",
    ):
        return self.minimizer(
            tolerance=tolerance, verbose=verbose, hessian=hessian,
        ).fit_multistart(
            n_starts=n_starts,
            seed=seed,
            include_default=include_default,
            simplex=simplex,
            strategy=strategy,
        )

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

    def fit_fraction_errors(self, result, *, acceptance_weighted=False):
        """Delta-method standard errors for print_fit_fractions()'s central values.

        Propagates the *joint* postfit covariance through both charges' fit
        fractions in a single Jacobian, not two independent ones -- plus_model
        and minus_model share almost every fit parameter (every CPRealImag
        coefficient and every dynamics parameter), so their fit fractions are
        correlated. That correlation is exactly what "mean" below needs:
        Var(mean) = 0.25*(Var(plus)+Var(minus)+2*Cov(plus,minus)), not a naive
        quadrature sum of the two charges' errors (see docs/cp_coefficients.md
        on why B+/B- are not independently normalized here). See
        dalitzplotfitter.observables.delta_method_errors for the propagation
        itself.
        """
        values = self.result_values(result)
        plus_cache = self.plus_model._fraction_cache(None, self.plus_efficiency if acceptance_weighted else None)
        minus_cache = self.minus_model._fraction_cache(None, self.minus_efficiency if acceptance_weighted else None)
        plus_names = [component.name for component in plus_cache.components]
        minus_names = [component.name for component in minus_cache.components]
        if plus_names != minus_names:
            raise ValueError("plus_model and minus_model must declare the same components in the same order")

        parameter_names = sorted({
            parameter.name
            for parameter in (*self.plus_model.parameters, *self.minus_model.parameters)
            if not parameter.fixed
        })

        # Differentiate each charge's integration graph only for its own rows.
        # Stack columns in the SAME union-of-parameters order, then propagate
        # the joint covariance once, including every cross-charge term.
        jacobian = jnp.concatenate([
            self.plus_model._fraction_jacobian(plus_cache, values, parameter_names),
            self.minus_model._fraction_jacobian(minus_cache, values, parameter_names),
        ])
        parameter_covariance = _covariance_matrix(result.covariance, parameter_names)
        covariance = np.asarray(jacobian @ parameter_covariance @ jacobian.T)
        n = len(plus_names)
        variance_plus = np.clip(np.diag(covariance)[:n], 0.0, None)
        variance_minus = np.clip(np.diag(covariance)[n:], 0.0, None)
        cross = np.diag(covariance[:n, n:])
        variance_mean = np.clip(0.25 * (variance_plus + variance_minus + 2.0 * cross), 0.0, None)

        return {
            "plus": dict(zip(plus_names, (float(v) for v in np.sqrt(variance_plus)))),
            "minus": dict(zip(minus_names, (float(v) for v in np.sqrt(variance_minus)))),
            "mean": dict(zip(plus_names, (float(v) for v in np.sqrt(variance_mean)))),
        }

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
        from dalitzplotfitter.workflow import _scaled_projection_weights

        total_events = self.plus_data.size + self.minus_data.size
        standalone = False
        if self.extended:
            plus_yield, minus_yield, standalone = _signal_yield_pair(self.signal_yield, values)
            plus_yield, minus_yield = float(plus_yield), float(minus_yield)
        elif self.background_categories:
            plus_yield = minus_yield = total_events * float(_resolve(self.signal_fraction, values))
        else:
            plus_yield = minus_yield = float(total_events)
        plus_w, minus_w = np.zeros(plus_sample.size), np.zeros(minus_sample.size)
        if plus_yield or minus_yield:
            _, integral_plus = self.plus_cache.evaluate(values)
            _, integral_minus = self.minus_cache.evaluate(values)
            norm = float(integral_plus + integral_minus)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("signal projection requires positive finite joint integral")
            for sample, model, efficiency, veto, integral, signal_yield_value, target in (
                (plus_sample, self.plus_model, self.plus_efficiency, self.plus_veto, integral_plus, plus_yield, plus_w),
                (minus_sample, self.minus_model, self.minus_efficiency, self.minus_veto, integral_minus, minus_yield, minus_w),
            ):
                # A YieldAsymmetry's n_plus/n_minus are already literal per-charge
                # counts (see YieldAsymmetry docstring); a shared yield still
                # needs the amplitude-driven integral_q/norm split.
                scale = signal_yield_value if standalone else signal_yield_value * float(integral) / norm
                if scale:
                    density = _acceptance(efficiency, veto, sample.as_dict()) * model.intensity(sample.as_dict(), values)
                    target[:] = _scaled_projection_weights(sample, density, scale)
        plus_components = [("signal", plus_sample, plus_w)]
        minus_components = [("signal", minus_sample, minus_w)]
        if not self.background_categories:
            return plus_components, minus_components
        if self.extended:
            bg_scales = [float(_resolve(c.yield_, values)) for c in self.background_categories]
        else:
            bg_total = total_events * (1.0 - float(_resolve(self.signal_fraction, values)))
            bw = np.asarray(self.base_objective.background_weights(values), dtype=float)
            bg_scales = [bg_total * float(w) for w in bw]
        for source, category, scale in zip(self.backgrounds, self.background_categories, bg_scales):
            if scale == 0:
                continue
            if not isinstance(source, CPBackgroundSpec):
                raise ValueError("plotting a precomputed CP background requires a CPBackgroundSpec with evaluable shapes")
            pr = jnp.asarray(source.plus_shape(plus_sample.as_dict())) if float(category.plus_probability) else jnp.zeros(plus_sample.size)
            mr = jnp.asarray(source.resolved_minus_shape(minus_sample.as_dict())) if float(category.minus_probability) else jnp.zeros(minus_sample.size)
            if source.apply_veto:
                if self.plus_veto is not None:
                    pr *= jnp.asarray(self.plus_veto(plus_sample.as_dict()))
                if self.minus_veto is not None:
                    mr *= jnp.asarray(self.minus_veto(minus_sample.as_dict()))
            pw = _scaled_projection_weights(plus_sample, pr, scale * float(category.plus_probability))
            mw = _scaled_projection_weights(minus_sample, mr, scale * float(category.minus_probability))
            plus_components.append((category.name, plus_sample, pw))
            minus_components.append((category.name, minus_sample, mw))
        return plus_components, minus_components

    def _projection_components(self, values, charge: str):
        """Compatibility wrapper using deterministic normalization samples."""
        if charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")
        plus, minus = self._projection_components_pair(
            values,
            self.plus_model.normalization_sample,
            self.minus_model.normalization_sample,
        )
        return plus if charge == "plus" else minus

    def plot_projection(self, result, variable="s13", *, bins=60, range=None, show_components=True, show_pulls=False, log_scale=False, projection_size=250_000, projection_seed=20260901, folded=False, partner_variable=None, fold_side="low", axes=None):
        """Plot smooth B+/B- projections without histogramming quadrature nodes.

        Two weighted phase-space MC samples are used only for rendering. Their
        component weights are normalized jointly across charges, preserving the
        integrated charge asymmetry of the fitted model.

        ``folded=True`` projects onto ``s_low = min(variable, partner_variable)``
        (``fold_side="low"``, default) or ``s_high = max(...)``
        (``fold_side="high"``), event by event, *within each charge's own
        subplot* (B+ and B- are never mixed); see
        ``FitSession.plot_projection`` for the identical-daughter convention
        this exploits.

        ``show_pulls=True`` adds a ``(observed-expected)/sqrt(expected)`` panel
        below each charge's histogram, sharing that column's x axis. It builds
        its own 2x2 figure and therefore requires ``axes=None``; the return
        value is then the full 2x2 axes grid (row 0 the histograms, row 1 the
        pulls) instead of the usual length-2 list.
        """
        import matplotlib.pyplot as plt
        if folded and partner_variable is None:
            raise ValueError("folded=True requires partner_variable")
        if fold_side not in ("low", "high"):
            raise ValueError("fold_side must be 'low' or 'high'")
        if show_pulls and axes is not None:
            raise ValueError(
                "show_pulls=True builds its own figure layout; pass axes=None"
            )
        fold_fn = np.minimum if fold_side == "low" else np.maximum

        def _folded_values(sample):
            values_ = np.asarray(getattr(sample, variable))
            if not folded:
                return values_
            partner_values = np.asarray(getattr(sample, partner_variable))
            return fold_fn(values_, partner_values)

        values = self.result_values(result)
        combined = np.concatenate([
            _folded_values(d) for d in (self.plus_data, self.minus_data)
        ])
        if range is None and combined.size == 0:
            raise ValueError("provide range when both charge datasets are empty")
        hist_range = range if range is not None else (float(np.min(combined)), float(np.max(combined)))
        edges = np.histogram_bin_edges(combined, bins=bins, range=hist_range)
        grid = None
        pulls_axes = (None, None)
        if axes is None:
            if show_pulls:
                _, grid = plt.subplots(
                    2, 2, figsize=(12, 7.2), sharex="col",
                    gridspec_kw={"height_ratios": (3, 1)},
                    constrained_layout=True,
                )
                axes, pulls_axes = grid[0], grid[1]
            else:
                _, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        plus_sample = self.plus_model.generate_phase_space(projection_size, seed=projection_seed)
        minus_sample = self.minus_model.generate_phase_space(projection_size, seed=projection_seed + 1)
        plus_components, minus_components = self._projection_components_pair(values, plus_sample, minus_sample)
        label = (
            rf"$s_{{\mathrm{{{fold_side}}}}}$" if folded else rf"${variable}$"
        )
        for ax, ax_pulls, charge, data, components in zip(
            axes, pulls_axes, ("plus", "minus"),
            (self.plus_data, self.minus_data), (plus_components, minus_components),
        ):
            dv = _folded_values(data)
            unit = r"GeV$^2$" if variable in ("s12","s13","s23") else ""
            _, observed, _, _ = plot_binned_data(
                dv, bins=edges, ax=ax,
                label=f"B{'+' if charge=='plus' else '-'} data",
                unit=unit, log_scale=log_scale,
            )
            total = np.zeros(bins)
            for name, sample, weights in components:
                cv = _folded_values(sample)
                counts, _ = np.histogram(cv, bins=edges, weights=np.asarray(weights))
                total += counts
                if show_components:
                    ax.stairs(counts, edges, label=name)
            ax.stairs(total, edges, label="total fit", linewidth=2.0)
            axis_label = label + (" [GeV$^2$]" if unit else "")
            ax.legend()
            if ax_pulls is None:
                ax.set_xlabel(axis_label)
                continue
            occupied = total > 0
            pulls = np.full(bins, np.nan)
            pulls[occupied] = (
                (observed[occupied] - total[occupied]) / np.sqrt(total[occupied])
            )
            _draw_pulls_1d(ax_pulls, edges, pulls)
            ax_pulls.set_xlabel(axis_label)
        return grid if show_pulls else axes

    def _projection_signal_density(self, sample, values, charge):
        """Normalized per-charge signal density at arbitrary points.

        Unlike ``FitSession``, ``CPFitSession`` has no compact-cache-optimized
        projection path; this mirrors what ``_projection_components_pair``
        already does to render MC projections (``model.intensity`` evaluated
        directly), just normalized by the charge's own fitted integral so the
        result integrates to one over that charge's Dalitz plane alone.
        """

        if charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")
        model = self.plus_model if charge == "plus" else self.minus_model
        cache = self.plus_cache if charge == "plus" else self.minus_cache
        efficiency = self.plus_efficiency if charge == "plus" else self.minus_efficiency
        veto = self.plus_veto if charge == "plus" else self.minus_veto
        _, integral = cache.evaluate(values)
        acceptance = _acceptance(efficiency, veto, sample.as_dict())
        return acceptance * model.intensity(sample.as_dict(), values) / integral

    def _total_density(self, sample, values, charge):
        """Fraction-weighted, unit-integral total fitted density f0(x|charge).

        Mirrors ``_projection_components_pair``'s extended/signal_fraction/
        plain branch structure and its amplitude-driven charge split
        (``integral_q/(integral_plus+integral_minus)``), but returns
        *fractions* (summing to one, conditioned on this one charge) rather
        than absolute event-count scales -- what the point-to-point
        dissimilarity test's ``f0`` requires. See
        ``FitSession._total_density`` for why this cannot reuse
        ``_scaled_projection_weights`` directly.
        """

        if charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")
        total_events = self.plus_data.size + self.minus_data.size
        standalone = False
        if self.extended:
            plus_yield, minus_yield, standalone = _signal_yield_pair(self.signal_yield, values)
            signal_yield_nominal = float(plus_yield if charge == "plus" else minus_yield)
        elif self.background_categories:
            signal_yield_nominal = total_events * float(_resolve(self.signal_fraction, values))
        else:
            signal_yield_nominal = float(total_events)

        signal_scale = 0.0
        if signal_yield_nominal:
            if standalone:
                signal_scale = signal_yield_nominal
            else:
                _, integral_plus = self.plus_cache.evaluate(values)
                _, integral_minus = self.minus_cache.evaluate(values)
                norm = float(integral_plus + integral_minus)
                integral = integral_plus if charge == "plus" else integral_minus
                signal_scale = signal_yield_nominal * float(integral) / norm

        if self.extended:
            bg_scales_total = [
                float(_resolve(category.yield_, values))
                for category in self.background_categories
            ]
        elif self.background_categories:
            bg_total = total_events * (1.0 - float(_resolve(self.signal_fraction, values)))
            weights = np.asarray(self.base_objective.background_weights(values), dtype=float)
            bg_scales_total = [bg_total * float(weight) for weight in weights]
        else:
            bg_scales_total = []

        probability_attr = "plus_probability" if charge == "plus" else "minus_probability"
        bg_scales = [
            scale * float(getattr(category, probability_attr))
            for scale, category in zip(bg_scales_total, self.background_categories)
        ]

        total_scale = signal_scale + sum(bg_scales)
        if total_scale <= 0:
            raise ValueError("total expected yield for this charge must be positive")

        density = (signal_scale / total_scale) * self._projection_signal_density(
            sample, values, charge
        )
        veto = self.plus_veto if charge == "plus" else self.minus_veto
        normalization_attr = (
            "plus_normalization" if charge == "plus" else "minus_normalization"
        )
        for source, category, scale in zip(
            self.backgrounds, self.background_categories, bg_scales
        ):
            if scale == 0:
                continue
            if not isinstance(source, CPBackgroundSpec):
                raise ValueError(
                    "goodness-of-fit density requires a CPBackgroundSpec with "
                    "evaluable shapes"
                )
            shape = source.plus_shape if charge == "plus" else source.resolved_minus_shape
            raw = jnp.asarray(shape(sample.as_dict()))
            if source.apply_veto and veto is not None:
                raw = raw * jnp.asarray(veto(sample.as_dict()))
            normalization = getattr(category, normalization_attr)
            density = density + (scale / total_scale) * (raw / normalization)
        return density

    def _default_free_parameters(self) -> int:
        return sum(1 for parameter in self.parameters if not parameter.fixed)

    def goodness_of_fit_projection(
        self,
        result,
        variable="s13",
        *,
        charge=None,
        bins=60,
        range=None,
        folded=False,
        partner_variable=None,
        fold_side="low",
        projection_size=250_000,
        projection_seed=20260901,
        n_free_parameters=None,
    ):
        """Binned Pearson chi2 goodness-of-fit test on a 1D projection.

        Returns a ``{"plus": ..., "minus": ...}`` dict of
        :class:`~dalitzplotfitter.goodness_of_fit.BinnedChi2Result` by
        default, or a single result when ``charge`` is given. Uses the same
        reweighted-MC-projection histogram as ``plot_projection`` for the
        expected counts.
        """

        if folded and partner_variable is None:
            raise ValueError("folded=True requires partner_variable")
        if fold_side not in ("low", "high"):
            raise ValueError("fold_side must be 'low' or 'high'")
        if charge is not None and charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")
        fold_fn = np.minimum if fold_side == "low" else np.maximum

        def _folded_values(sample):
            values_ = np.asarray(getattr(sample, variable))
            if not folded:
                return values_
            partner_values = np.asarray(getattr(sample, partner_variable))
            return fold_fn(values_, partner_values)

        values = self.result_values(result)
        combined = np.concatenate(
            [_folded_values(d) for d in (self.plus_data, self.minus_data)]
        )
        if range is None and combined.size == 0:
            raise ValueError("provide range when both charge datasets are empty")
        hist_range = range if range is not None else (
            float(np.min(combined)), float(np.max(combined))
        )
        edges = np.histogram_bin_edges(combined, bins=bins, range=hist_range)

        plus_sample = self.plus_model.generate_phase_space(projection_size, seed=projection_seed)
        minus_sample = self.minus_model.generate_phase_space(
            projection_size, seed=projection_seed + 1
        )
        plus_components, minus_components = self._projection_components_pair(
            values, plus_sample, minus_sample
        )

        if n_free_parameters is None:
            n_free_parameters = self._default_free_parameters()

        results = {}
        for name, data, components in (
            ("plus", self.plus_data, plus_components),
            ("minus", self.minus_data, minus_components),
        ):
            observed, _ = np.histogram(_folded_values(data), bins=edges)
            expected = np.zeros(len(edges) - 1, dtype=float)
            for _, component_sample, weights in components:
                component_values = _folded_values(component_sample)
                counts, _ = np.histogram(
                    component_values, bins=edges, weights=np.asarray(weights)
                )
                expected += counts
            results[name] = chi2_from_histograms(
                observed, expected, n_free_parameters=n_free_parameters, edges=(edges,)
            )
        return results[charge] if charge is not None else results

    def goodness_of_fit_chi2(
        self,
        result,
        x="s13",
        y="s23",
        *,
        charge=None,
        bins=25,
        range=None,
        folded=False,
        square_dalitz=False,
        mother_mass=None,
        masses=None,
        pair=(0, 1),
        projection_size=250_000,
        projection_seed=20260901,
        n_free_parameters=None,
    ):
        """Binned Pearson chi2 goodness-of-fit test on the Dalitz plane.

        Returns a ``{"plus": ..., "minus": ...}`` dict of
        :class:`~dalitzplotfitter.goodness_of_fit.BinnedChi2Result` by
        default, or a single result when ``charge`` is given. See
        ``FitSession.goodness_of_fit_chi2`` for the ``square_dalitz``/
        ``folded`` conventions.
        """

        if square_dalitz and (mother_mass is None or masses is None):
            raise ValueError("square_dalitz=True requires mother_mass and masses")
        if charge is not None and charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")

        def _coordinates(sample):
            if square_dalitz:
                data = sample.as_dict()
                mp, tp = invariants_to_square_dalitz(
                    data["s12"], data["s13"], data["s23"],
                    mother_mass=mother_mass,
                    masses=masses,
                    pair=pair,
                )
                if folded:
                    tp = fold_thetaprime(tp)
                return np.asarray(mp), np.asarray(tp)
            x_values = np.asarray(getattr(sample, x))
            y_values = np.asarray(getattr(sample, y))
            if folded:
                x_values, y_values = (
                    np.minimum(x_values, y_values),
                    np.maximum(x_values, y_values),
                )
            return x_values, y_values

        if range is not None:
            hist_range = range
        elif square_dalitz:
            hist_range = ((0.0, 1.0), (0.0, 0.5) if folded else (0.0, 1.0))
        else:
            hist_range = None

        values = self.result_values(result)
        plus_sample = self.plus_model.generate_phase_space(
            projection_size, seed=projection_seed
        )
        minus_sample = self.minus_model.generate_phase_space(
            projection_size, seed=projection_seed + 1
        )
        plus_components, minus_components = self._projection_components_pair(
            values, plus_sample, minus_sample
        )

        if n_free_parameters is None:
            n_free_parameters = self._default_free_parameters()

        results = {}
        for name, data, components in (
            ("plus", self.plus_data, plus_components),
            ("minus", self.minus_data, minus_components),
        ):
            data_x, data_y = _coordinates(data)
            observed, x_edges, y_edges = np.histogram2d(
                data_x, data_y, bins=bins, range=hist_range
            )
            expected = np.zeros_like(observed)
            for _, component_sample, weights in components:
                component_x, component_y = _coordinates(component_sample)
                counts, _, _ = np.histogram2d(
                    component_x,
                    component_y,
                    bins=[x_edges, y_edges],
                    weights=np.asarray(weights),
                )
                expected += counts
            results[name] = chi2_from_histograms(
                observed,
                expected,
                n_free_parameters=n_free_parameters,
                edges=(x_edges, y_edges),
            )
        return results[charge] if charge is not None else results

    def point_to_point_dissimilarity(
        self,
        result,
        *,
        charge,
        x="s13",
        y="s23",
        sigma_bar=0.01,
        mc_size=None,
        n_permutations=200,
        seed=20260901,
        max_total_events=5_000,
        candidate_pool_size=200_000,
    ):
        """Unbinned point-to-point dissimilarity test for one charge.

        Run once per charge (``charge="plus"``/``"minus"``): the underlying
        statistic (Williams, arXiv:1006.3019) compares one data sample
        against one reference sample from one density, so a "joint" PPD
        across both charges is not a documented statistic and is not
        offered here. See ``FitSession.point_to_point_dissimilarity``.
        """

        if charge not in ("plus", "minus"):
            raise ValueError("charge must be 'plus' or 'minus'")
        values = self.result_values(result)
        data = self.plus_data if charge == "plus" else self.minus_data
        model = self.plus_model if charge == "plus" else self.minus_model
        n_data = data.size
        if mc_size is None:
            mc_size = min(10 * n_data, max(max_total_events - n_data, 1))

        candidate = model.generate_phase_space(candidate_pool_size, seed=seed)
        candidate_density = self._total_density(candidate, values, charge)
        target_weights = candidate.weights * candidate_density
        replica = weighted_resample(
            jax.random.PRNGKey(seed), candidate, target_weights, mc_size
        )

        data_density = self._total_density(data, values, charge)
        replica_density = self._total_density(replica, values, charge)
        phase_space_area = float(jnp.mean(model.normalization_sample.weights))

        data_xy = np.column_stack(
            [np.asarray(getattr(data, x)), np.asarray(getattr(data, y))]
        )
        replica_xy = np.column_stack(
            [np.asarray(getattr(replica, x)), np.asarray(getattr(replica, y))]
        )

        return _point_to_point_dissimilarity(
            data_xy,
            replica_xy,
            np.asarray(data_density),
            np.asarray(replica_density),
            sigma_bar=sigma_bar,
            phase_space_area=phase_space_area,
            n_permutations=n_permutations,
            seed=seed + 1,
            max_total_events=max_total_events,
        )


__all__ = ["CPBackgroundSpec", "CPFitSession"]
