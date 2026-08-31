"""Repeated generate-and-fit studies for Dalitz-plot models.

The :class:`GenFit` driver implements the standard pseudoexperiment workflow:
generate independent toys from one injected model, fit every toy, collect the
fitted parameters and fit diagnostics, and summarize their distributions with
Gaussian fits.

The first implementation is intentionally optimized for coefficient-only fits.
For that common closure-test case the expensive amplitude evaluation on the
candidate pool and the normalization matrix are cached once and reused by all
pseudoexperiments.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit

from dalitzplotfitter.fit.parameters import Parameter, ParameterKind
from dalitzplotfitter.integration import matrix_normalization
from dalitzplotfitter.kinematics import DalitzGrid, PhaseSpaceSample


@dataclass(frozen=True)
class GaussianFitResult:
    """Gaussian maximum-likelihood fit to one GenFit distribution."""

    mean: float
    sigma: float
    mean_error: float
    sigma_error: float
    valid: bool
    n_entries: int


@dataclass(frozen=True)
class GenFitRecord:
    """Result of one generated pseudoexperiment and one fit."""

    index: int
    valid: bool
    values: dict[str, float]
    errors: dict[str, float]
    start_values: dict[str, float]
    nll: float
    truth_nll: float
    edm: float
    nfcn: int


class GenFitResult:
    """Collection, plotting helpers, and statistics for a GenFit study."""

    def __init__(
        self,
        records: tuple[GenFitRecord, ...],
        truth_values: Mapping[str, float],
        parameter_names: tuple[str, ...],
    ):
        self.records = records
        self.truth_values = {name: float(value) for name, value in truth_values.items()}
        self.parameter_names = tuple(parameter_names)
        self._gaussian_cache: dict[str, GaussianFitResult] = {}

    @property
    def n_fits(self) -> int:
        return len(self.records)

    @property
    def n_valid(self) -> int:
        return sum(record.valid for record in self.records)

    @property
    def success_rate(self) -> float:
        if not self.records:
            return math.nan
        return self.n_valid / self.n_fits

    @property
    def valid_mask(self) -> np.ndarray:
        return np.asarray([record.valid for record in self.records], dtype=bool)

    @property
    def nll(self) -> np.ndarray:
        return np.asarray([record.nll for record in self.records], dtype=float)

    @property
    def truth_nll(self) -> np.ndarray:
        return np.asarray([record.truth_nll for record in self.records], dtype=float)

    @property
    def edm(self) -> np.ndarray:
        return np.asarray([record.edm for record in self.records], dtype=float)

    @property
    def nfcn(self) -> np.ndarray:
        return np.asarray([record.nfcn for record in self.records], dtype=int)

    def values(self, name: str, *, valid_only: bool = True) -> np.ndarray:
        self._check_parameter(name)
        values = np.asarray([record.values[name] for record in self.records], dtype=float)
        return values[self.valid_mask] if valid_only else values

    def errors(self, name: str, *, valid_only: bool = True) -> np.ndarray:
        self._check_parameter(name)
        values = np.asarray([record.errors[name] for record in self.records], dtype=float)
        return values[self.valid_mask] if valid_only else values

    def starts(self, name: str) -> np.ndarray:
        self._check_parameter(name)
        return np.asarray([record.start_values[name] for record in self.records], dtype=float)

    def pulls(self, name: str, *, valid_only: bool = True) -> np.ndarray:
        truth = self.truth_values[name]
        values = self.values(name, valid_only=valid_only)
        errors = self.errors(name, valid_only=valid_only)
        with np.errstate(divide="ignore", invalid="ignore"):
            pulls = (values - truth) / errors
        return pulls[np.isfinite(pulls)]

    def _check_parameter(self, name: str) -> None:
        if name not in self.parameter_names:
            raise KeyError(f"Unknown GenFit parameter {name!r}")

    @staticmethod
    def _gaussian_fit(values: np.ndarray) -> GaussianFitResult:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            return GaussianFitResult(
                mean=float(values[0]) if values.size else math.nan,
                sigma=math.nan,
                mean_error=math.nan,
                sigma_error=math.nan,
                valid=False,
                n_entries=int(values.size),
            )

        mean0 = float(np.mean(values))
        sigma0 = float(np.std(values, ddof=1))
        if not np.isfinite(sigma0) or sigma0 <= 0.0:
            sigma0 = max(abs(mean0), 1.0) * 1e-6

        def gaussian_nll(mean, sigma):
            if sigma <= 0.0:
                return np.inf
            z = (values - mean) / sigma
            return float(values.size * np.log(sigma) + 0.5 * np.sum(z * z))

        fit = Minuit(gaussian_nll, mean=mean0, sigma=sigma0)
        fit.errordef = 0.5
        fit.limits["sigma"] = (np.finfo(float).tiny, None)
        fit.migrad()
        fit.hesse()
        return GaussianFitResult(
            mean=float(fit.values["mean"]),
            sigma=float(fit.values["sigma"]),
            mean_error=float(fit.errors["mean"]),
            sigma_error=float(fit.errors["sigma"]),
            valid=bool(fit.valid),
            n_entries=int(values.size),
        )

    def gaussian_fit(self, name: str) -> GaussianFitResult:
        """Fit a Gaussian to a fitted-parameter distribution or to ``'nll'``."""

        if name not in self._gaussian_cache:
            if name == "nll":
                values = self.nll[self.valid_mask]
            else:
                values = self.values(name, valid_only=True)
            self._gaussian_cache[name] = self._gaussian_fit(values)
        return self._gaussian_cache[name]

    def summary(self) -> list[dict[str, float | int | str | bool]]:
        """Return one statistical-summary row per parameter plus the NLL."""

        rows: list[dict[str, float | int | str | bool]] = []
        for name in (*self.parameter_names, "nll"):
            values = self.nll[self.valid_mask] if name == "nll" else self.values(name)
            gaussian = self.gaussian_fit(name)
            rows.append(
                {
                    "name": name,
                    "entries": int(values.size),
                    "sample_mean": float(np.mean(values)) if values.size else math.nan,
                    "sample_std": float(np.std(values, ddof=1)) if values.size > 1 else math.nan,
                    "gauss_mean": gaussian.mean,
                    "gauss_mean_error": gaussian.mean_error,
                    "gauss_sigma": gaussian.sigma,
                    "gauss_sigma_error": gaussian.sigma_error,
                    "gauss_valid": gaussian.valid,
                }
            )
        return rows

    def print_summary(self) -> None:
        """Print convergence and Gaussian-summary tables."""

        print(
            f"GenFit: {self.n_valid}/{self.n_fits} valid fits "
            f"({100.0 * self.success_rate:.1f}%)"
        )
        print(
            f"{'quantity':18s} {'mean':>13s} {'std':>13s} "
            f"{'gauss mean':>13s} {'gauss sigma':>13s}"
        )
        for row in self.summary():
            print(
                f"{str(row['name']):18s} "
                f"{float(row['sample_mean']):13.6g} "
                f"{float(row['sample_std']):13.6g} "
                f"{float(row['gauss_mean']):13.6g} "
                f"{float(row['gauss_sigma']):13.6g}"
            )

    def plot(
        self,
        name: str,
        *,
        bins: int | str = "auto",
        ax=None,
        density: bool = False,
    ):
        """Plot one GenFit histogram and overlay its fitted Gaussian."""

        if name == "nll":
            values = self.nll[self.valid_mask]
            truth = None
        else:
            values = self.values(name)
            truth = self.truth_values[name]
        gaussian = self.gaussian_fit(name)

        if ax is None:
            _, ax = plt.subplots(figsize=(7.0, 4.8))
        counts, edges, _ = ax.hist(values, bins=bins, density=density, alpha=0.65)
        centers = 0.5 * (edges[:-1] + edges[1:])
        if gaussian.valid and gaussian.sigma > 0.0 and centers.size:
            x = np.linspace(edges[0], edges[-1], 500)
            pdf = np.exp(-0.5 * ((x - gaussian.mean) / gaussian.sigma) ** 2)
            if density:
                y = pdf / (math.sqrt(2.0 * math.pi) * gaussian.sigma)
            else:
                bin_width = float(np.mean(np.diff(edges)))
                y = (
                    gaussian.n_entries
                    * bin_width
                    * pdf
                    / (math.sqrt(2.0 * math.pi) * gaussian.sigma)
                )
            ax.plot(x, y, label="Gaussian fit")
        if truth is not None:
            ax.axvline(truth, linestyle="--", label="truth")
        ax.set_xlabel(name)
        ax.set_ylabel("Density" if density else "Pseudoexperiments")
        ax.set_title(f"GenFit distribution: {name}")
        ax.legend()
        return ax

    def plot_all(self, *, bins: int | str = "auto") -> dict[str, object]:
        """Create one histogram figure for every fitted parameter and the NLL."""

        figures = {}
        for name in (*self.parameter_names, "nll"):
            fig, ax = plt.subplots(figsize=(7.0, 4.8))
            self.plot(name, bins=bins, ax=ax)
            fig.tight_layout()
            figures[name] = fig
        return figures


class GenFit:
    """Run repeated generated-sample fits for a coefficient-only Dalitz model.

    Parameters
    ----------
    model:
        Configured :class:`~dalitzplotfitter.decay.DecayModel`.
    n_fits:
        Number of pseudoexperiments.
    sample_size:
        Number of unweighted events in each pseudoexperiment.
    truth_values:
        Injected free-parameter values. If omitted, the current values of the
        model's free parameters are used.

    Notes
    -----
    Floating dynamics parameters are deliberately rejected in this first
    implementation. Coefficient-only GenFits can reuse the cached candidate-pool
    amplitudes and fixed normalization matrix and are therefore much faster for
    hundreds of pseudoexperiments.
    """

    def __init__(
        self,
        model,
        n_fits: int,
        sample_size: int,
        *,
        truth_values: Mapping[str, float] | None = None,
        normalization_sample: PhaseSpaceSample | None = None,
        grid_resolution: int = 1000,
        pool_size: int = 1_000_000,
        seed: int = 791,
        pool_seed: int = 2000,
        start_range: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
        simplex: bool = False,
        ncall: int | None = 100_000,
        tolerance: float = 1e-4,
        verbose: int = 1,
    ):
        if isinstance(n_fits, bool) or not isinstance(n_fits, int) or n_fits <= 0:
            raise ValueError("n_fits must be a positive integer")
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")
        if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size <= 0:
            raise ValueError("pool_size must be a positive integer")
        if pool_size < sample_size:
            raise ValueError("pool_size must be at least sample_size")
        if grid_resolution <= 0:
            raise ValueError("grid_resolution must be positive")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if ncall is not None and (isinstance(ncall, bool) or not isinstance(ncall, int) or ncall <= 0):
            raise ValueError("ncall must be a positive integer or None")
        if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer")

        self.model = model
        self.n_fits = n_fits
        self.sample_size = sample_size
        self.grid_resolution = grid_resolution
        self.pool_size = pool_size
        self.seed = int(seed)
        self.pool_seed = int(pool_seed)
        self.start_range = start_range
        self.simplex = bool(simplex)
        self.ncall = ncall
        self.tolerance = float(tolerance)
        self.verbose = verbose

        self.free_parameters = tuple(p for p in model.parameters if not p.fixed)
        if not self.free_parameters:
            raise ValueError("GenFit requires at least one free parameter")
        floating_dynamics = tuple(
            p for p in self.free_parameters if p.kind is ParameterKind.DYNAMICS
        )
        if floating_dynamics:
            names = ", ".join(p.name for p in floating_dynamics)
            raise NotImplementedError(
                "GenFit currently supports coefficient-only fits; floating dynamics "
                f"parameters were found: {names}"
            )
        unsupported = tuple(
            p for p in self.free_parameters if p.kind is not ParameterKind.COEFFICIENT
        )
        if unsupported:
            names = ", ".join(p.name for p in unsupported)
            raise NotImplementedError(
                "GenFit currently supports free coefficient parameters only; "
                f"unsupported free parameters were found: {names}"
            )

        if truth_values is None:
            truth_values = {p.name: float(p.value) for p in self.free_parameters}
        missing = [p.name for p in self.free_parameters if p.name not in truth_values]
        if missing:
            raise ValueError(
                "truth_values is missing free parameters: " + ", ".join(missing)
            )
        known = {p.name for p in model.parameters}
        unknown = set(truth_values) - known
        if unknown:
            raise ValueError(
                "truth_values contains unknown parameters: " + ", ".join(sorted(unknown))
            )
        self.truth_values = {
            p.name: float(truth_values[p.name]) for p in self.free_parameters
        }

        if normalization_sample is None:
            normalization_sample = DalitzGrid(
                model.channel.parent_mass,
                model.channel.daughter_masses,
                resolution=grid_resolution,
            ).sample()
        self.normalization_sample = normalization_sample

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[GenFit] {message}", flush=True)

    def _draw_start(
        self,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        starts = {}
        for parameter in self.free_parameters:
            if self.start_range is None:
                starts[parameter.name] = self._draw_parameter(parameter, rng)
                continue
            if isinstance(self.start_range, Mapping):
                low, high = self.start_range[parameter.name]
            else:
                low, high = self.start_range
            value = float(rng.uniform(low, high))
            if parameter.bounds is not None:
                bound_low, bound_high = parameter.bounds
                if bound_low is not None:
                    value = max(value, float(bound_low))
                if bound_high is not None:
                    value = min(value, float(bound_high))
            starts[parameter.name] = value
        return starts

    @staticmethod
    def _draw_parameter(parameter: Parameter, rng: np.random.Generator) -> float:
        if parameter.bounds is not None:
            low, high = parameter.bounds
            if low is not None and high is not None:
                return float(rng.uniform(low, high))
        scale = (
            10.0 * parameter.step
            if parameter.step is not None
            else max(abs(float(parameter.value)), 1.0) * 0.25
        )
        value = float(rng.normal(float(parameter.value), scale))
        if parameter.bounds is not None:
            low, high = parameter.bounds
            if low is not None:
                value = max(value, float(low))
            if high is not None:
                value = min(value, float(high))
        return value

    def run(self) -> GenFitResult:
        """Generate and fit all pseudoexperiments and return their collection."""

        self._log(
            f"preparing normalization and {self.pool_size:,}-event candidate pool"
        )
        pool = self.model.generate_phase_space(self.pool_size, seed=self.pool_seed)
        pool_cache = self.model.prepare_cache(pool, self.normalization_sample)
        truth_intensity, truth_normalization = pool_cache.evaluate(self.truth_values)
        target_weights = jnp.asarray(pool.weights * truth_intensity)
        probabilities = target_weights / jnp.sum(target_weights)

        names = tuple(parameter.name for parameter in self.free_parameters)
        fixed = {
            parameter.name: float(parameter.value)
            for parameter in self.model.parameters
            if parameter.fixed
        }
        components = pool_cache.components
        normalization_matrix = pool_cache.normalization_matrix_fixed
        sample_size = self.sample_size

        def vector_nll(vector, data_components):
            values = dict(fixed)
            values.update({name: vector[i] for i, name in enumerate(names)})
            coefficients = jnp.asarray(
                [component.coefficient.value(values) for component in components]
            )
            amplitude = data_components @ coefficients
            intensity = jnp.abs(amplitude) ** 2
            normalization = matrix_normalization(coefficients, normalization_matrix)
            return -jnp.sum(jnp.log(jnp.clip(intensity, min=1e-300))) + sample_size * jnp.log(normalization)

        value_and_grad = jax.jit(jax.value_and_grad(vector_nll, argnums=0))
        truth_vector = jnp.asarray([self.truth_values[name] for name in names])
        records: list[GenFitRecord] = []

        self._log(
            f"running {self.n_fits} pseudoexperiments with {self.sample_size:,} events each"
        )
        for index in range(self.n_fits):
            sample_key = jax.random.key(self.seed + index)
            indices = jax.random.choice(
                sample_key,
                pool.size,
                shape=(self.sample_size,),
                replace=True,
                p=probabilities,
            )
            data_components = pool_cache.data_components[indices]
            start_rng = np.random.default_rng(self.seed + 1_000_000 + index)
            start_values = self._draw_start(start_rng)
            start = tuple(start_values[name] for name in names)

            def fcn(*values):
                value, _ = value_and_grad(jnp.asarray(values), data_components)
                return float(value)

            def grad(*values):
                _, gradient = value_and_grad(jnp.asarray(values), data_components)
                return np.asarray(gradient, dtype=float)

            fit = Minuit(fcn, *start, name=names, grad=grad)
            fit.errordef = 0.5
            fit.tol = self.tolerance
            fit.strategy = 1
            fit.print_level = 0
            for parameter in self.free_parameters:
                if parameter.bounds is not None:
                    fit.limits[parameter.name] = parameter.bounds
                if parameter.step is not None:
                    fit.errors[parameter.name] = parameter.step
            if self.simplex:
                fit.simplex()
            fit.migrad(ncall=self.ncall)
            fit.strategy = 2
            fit.migrad(ncall=self.ncall)
            fit.hesse()

            truth_nll, _ = value_and_grad(truth_vector, data_components)
            valid = bool(fit.valid) and np.isfinite(float(fit.fval))
            record = GenFitRecord(
                index=index,
                valid=valid,
                values={name: float(fit.values[name]) for name in names},
                errors={name: float(fit.errors[name]) for name in names},
                start_values=start_values,
                nll=float(fit.fval),
                truth_nll=float(truth_nll),
                edm=float(fit.fmin.edm),
                nfcn=int(fit.nfcn),
            )
            records.append(record)
            if self.verbose >= 2 or (
                self.verbose == 1
                and ((index + 1) == 1 or (index + 1) % max(1, self.n_fits // 10) == 0)
            ):
                self._log(
                    f"{index + 1}/{self.n_fits}: valid={valid} "
                    f"NLL={record.nll:.6f} EDM={record.edm:.3e}"
                )

        result = GenFitResult(
            tuple(records),
            self.truth_values,
            names,
        )
        self._log(
            f"finished: {result.n_valid}/{result.n_fits} valid fits "
            f"({100.0 * result.success_rate:.1f}%)"
        )
        return result
