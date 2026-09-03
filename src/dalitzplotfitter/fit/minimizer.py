"""Minuit driver using JAX values and automatic gradients."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from .parameters import Parameter


# ``FitSession.fit()`` constructs a new Minimizer for each call. The objective
# object itself is cached by the session, so use its identity to retain the JAX
# executable across those short-lived Minimizer instances. The compiled closure
# keeps only a weak reference to the objective, so this global lookup cannot keep
# a finished fit session alive.
_SHARED_BACKENDS: dict[tuple[int, tuple], tuple[weakref.ReferenceType, object]] = {}


@dataclass(frozen=True)
class GradientCheckResult:
    """Comparison between JAX and central finite-difference gradients."""

    names: tuple[str, ...]
    point: dict[str, float]
    jax_gradient: np.ndarray
    finite_difference_gradient: np.ndarray
    absolute_error: np.ndarray
    relative_error: np.ndarray

    @property
    def max_absolute_error(self) -> float:
        return float(np.max(self.absolute_error))

    @property
    def max_relative_error(self) -> float:
        return float(np.max(self.relative_error))


@dataclass(frozen=True)
class MultiStartResult:
    """Collection of independent minimizations and the best valid minimum."""

    best: object
    results: tuple[object, ...]
    starts: tuple[dict[str, float], ...]

    @property
    def valid_results(self) -> tuple[object, ...]:
        return tuple(
            result
            for result in self.results
            if bool(result.valid) and np.isfinite(float(result.fval))
        )


class Minimizer:
    """Minimize a mapping-based JAX objective with iminuit.

    ``errordef=0.5`` is the Minuit convention for a negative log-likelihood.
    ``tolerance`` is passed directly to ``Minuit.tol``. The default ``1e-4``
    provides a stricter convergence target than iminuit's generic default while
    remaining numerically practical for the amplitude-fit likelihoods used here.

    The JAX value-and-gradient program is compiled lazily. It is cached both by
    the ``Minimizer`` instance and, while the objective remains alive, across
    different Minimizer instances that wrap the same objective and parameter
    layout. This means repeated ``FitSession.fit()`` or ``CPFitSession.fit()``
    calls reuse the already-compiled XLA executable instead of paying the JIT
    cost again.

    Value and gradient callbacks share a one-point host cache, so when Minuit
    asks for both at identical parameters the expensive device evaluation and
    device-to-host synchronization occur only once.

    ``fit`` exposes Minuit strategies 0, 1 and 2 directly. The historical
    project default remains strategy 2 with HESSE enabled; that path keeps the
    established two-MIGRAD refinement before HESSE. Strategies 0 and 1 use one
    MIGRAD pass, which is useful for faster preliminary fits of large models.
    """

    def __init__(
        self,
        objective: Callable,
        parameters: Sequence[Parameter],
        *,
        errordef: float = 0.5,
        tolerance: float = 1e-4,
        verbose: int = 0,
    ):
        if errordef <= 0:
            raise ValueError("errordef must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer")
        self.objective = objective
        self.parameters = tuple(parameters)
        self.errordef = float(errordef)
        self.tolerance = float(tolerance)
        self.verbose = int(verbose)
        self._backend_cache = None

    def _log(self, message: str) -> None:
        if self.verbose >= 1:
            print(f"[Minimizer] {message}", flush=True)

    @staticmethod
    def _summary(result) -> str:
        return (
            f"valid={bool(result.valid)}  "
            f"NLL={float(result.fval):.6f}  "
            f"EDM={float(result.fmin.edm):.3e}  "
            f"nfcn={int(result.nfcn)}"
        )

    def _backend_signature(self) -> tuple:
        return tuple(
            (
                parameter.name,
                bool(parameter.fixed),
                float(parameter.value) if parameter.fixed else None,
            )
            for parameter in self.parameters
        )

    def _shared_backend(self):
        key = (id(self.objective), self._backend_signature())
        cached = _SHARED_BACKENDS.get(key)
        if cached is None:
            return key, None
        objective_ref, backend = cached
        if objective_ref() is self.objective:
            return key, backend
        _SHARED_BACKENDS.pop(key, None)
        return key, None

    def _backend(self):
        """Return persistent compiled JAX and Minuit callbacks."""

        if self._backend_cache is not None:
            return self._backend_cache

        shared_key, shared = self._shared_backend()
        if shared is not None:
            self._backend_cache = shared
            return shared

        free = tuple(parameter for parameter in self.parameters if not parameter.fixed)
        fixed = {
            parameter.name: parameter.value
            for parameter in self.parameters
            if parameter.fixed
        }
        names = tuple(parameter.name for parameter in free)
        if not free:
            raise ValueError("At least one free parameter is required")

        try:
            objective_ref = weakref.ref(
                self.objective,
                lambda _reference, cache_key=shared_key: _SHARED_BACKENDS.pop(
                    cache_key, None
                ),
            )
        except TypeError:
            objective_ref = None
            objective = self.objective

        if objective_ref is not None:
            def vector_objective(vector):
                objective = objective_ref()
                if objective is None:
                    raise RuntimeError("fit objective was released before JAX evaluation")
                mapping = dict(fixed)
                mapping.update({name: vector[i] for i, name in enumerate(names)})
                return objective(mapping)
        else:
            def vector_objective(vector):
                mapping = dict(fixed)
                mapping.update({name: vector[i] for i, name in enumerate(names)})
                return objective(mapping)

        value_and_grad = jax.jit(jax.value_and_grad(vector_objective))

        last_point: np.ndarray | None = None
        last_value: float | None = None
        last_gradient: np.ndarray | None = None

        def evaluate(values):
            nonlocal last_point, last_value, last_gradient
            point = np.asarray(values, dtype=float)
            if last_point is not None and np.array_equal(point, last_point):
                return last_value, last_gradient

            value_device, gradient_device = value_and_grad(jnp.asarray(point))
            value_host, gradient_host = jax.device_get((value_device, gradient_device))
            last_point = point.copy()
            last_value = float(value_host)
            last_gradient = np.asarray(gradient_host, dtype=float)
            return last_value, last_gradient

        def fcn(*values):
            value, _ = evaluate(values)
            return value

        def grad(*values):
            _, gradient = evaluate(values)
            return gradient

        backend = (free, names, fcn, grad)
        self._backend_cache = backend
        if objective_ref is not None:
            _SHARED_BACKENDS[shared_key] = (objective_ref, backend)
        return backend

    def _validate_start_values(
        self,
        start_values: Mapping[str, float] | None,
    ) -> dict[str, float]:
        values = {} if start_values is None else dict(start_values)
        known_names = {parameter.name for parameter in self.parameters}
        unknown = set(values) - known_names
        if unknown:
            unknown_text = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown starting parameters: {unknown_text}")
        for parameter in self.parameters:
            if parameter.name not in values or parameter.bounds is None:
                continue
            value = float(values[parameter.name])
            low, high = parameter.bounds
            if low is not None and value < low:
                raise ValueError(
                    f"starting value {value} is below the lower bound {low} for "
                    f"{parameter.name!r}"
                )
            if high is not None and value > high:
                raise ValueError(
                    f"starting value {value} is above the upper bound {high} for "
                    f"{parameter.name!r}"
                )
        return values

    @staticmethod
    def _validate_ncall(ncall: int | None) -> int | None:
        if ncall is None:
            return None
        if isinstance(ncall, bool) or not isinstance(ncall, int) or ncall <= 0:
            raise ValueError("ncall must be a positive integer or None")
        return ncall

    @staticmethod
    def _validate_strategy(strategy: int) -> int:
        if isinstance(strategy, bool) or not isinstance(strategy, int) or strategy not in (0, 1, 2):
            raise ValueError("strategy must be one of 0, 1 or 2")
        return int(strategy)

    def check_gradient(
        self,
        values: Mapping[str, float] | None = None,
        *,
        step_scale: float = 1e-5,
        relative_floor: float = 1e-12,
        print_table: bool = True,
    ) -> GradientCheckResult:
        if step_scale <= 0:
            raise ValueError("step_scale must be positive")
        if relative_floor <= 0:
            raise ValueError("relative_floor must be positive")

        free, names, fcn, grad = self._backend()
        supplied = self._validate_start_values(values)
        point = np.asarray(
            [float(supplied.get(parameter.name, parameter.value)) for parameter in free],
            dtype=float,
        )

        jax_gradient = np.asarray(grad(*point), dtype=float)
        finite_difference = np.empty_like(point)

        for i, value in enumerate(point):
            step = step_scale * max(abs(float(value)), 1.0)
            plus = point.copy()
            minus = point.copy()
            plus[i] += step
            minus[i] -= step
            finite_difference[i] = (fcn(*plus) - fcn(*minus)) / (2.0 * step)

        absolute_error = np.abs(jax_gradient - finite_difference)
        scale = np.maximum(
            np.maximum(np.abs(jax_gradient), np.abs(finite_difference)),
            relative_floor,
        )
        relative_error = absolute_error / scale

        result = GradientCheckResult(
            names=names,
            point={name: float(point[i]) for i, name in enumerate(names)},
            jax_gradient=jax_gradient,
            finite_difference_gradient=finite_difference,
            absolute_error=absolute_error,
            relative_error=relative_error,
        )

        if print_table:
            print(
                f"{'parameter':18s} {'JAX':>14s} {'finite diff':>14s} "
                f"{'abs err':>12s} {'rel err':>12s}"
            )
            for i, name in enumerate(names):
                print(
                    f"{name:18s} {jax_gradient[i]:14.6e} "
                    f"{finite_difference[i]:14.6e} {absolute_error[i]:12.3e} "
                    f"{relative_error[i]:12.3e}"
                )
            print(
                f"max abs error = {result.max_absolute_error:.3e}\n"
                f"max rel error = {result.max_relative_error:.3e}"
            )

        return result

    def _run(
        self,
        free,
        names,
        fcn,
        grad,
        *,
        start_values: Mapping[str, float] | None,
        strategy: int,
        hesse: bool,
        simplex: bool,
        ncall: int | None = None,
    ):
        from iminuit import Minuit

        ncall = self._validate_ncall(ncall)
        strategy = self._validate_strategy(strategy)
        supplied = self._validate_start_values(start_values)
        start = tuple(
            float(supplied.get(parameter.name, parameter.value)) for parameter in free
        )
        minuit = Minuit(fcn, *start, name=names, grad=grad)
        minuit.errordef = self.errordef
        minuit.tol = self.tolerance
        minuit.strategy = strategy
        minuit.print_level = max(0, self.verbose - 1)
        for parameter in free:
            if parameter.bounds is not None:
                minuit.limits[parameter.name] = parameter.bounds
            if parameter.step is not None:
                minuit.errors[parameter.name] = parameter.step
        if simplex:
            minuit.simplex()
        minuit.migrad(ncall=ncall)
        if strategy == 2:
            minuit.migrad(ncall=ncall)
        if hesse:
            minuit.hesse()
        return minuit

    def fit(
        self,
        start_values: Mapping[str, float] | None = None,
        *,
        simplex: bool = False,
        ncall: int | None = None,
        strategy: int = 2,
        hesse: bool = True,
    ):
        ncall = self._validate_ncall(ncall)
        strategy = self._validate_strategy(strategy)
        if not isinstance(hesse, bool):
            raise ValueError("hesse must be a boolean")
        free, names, fcn, grad = self._backend()
        self._log(
            f"single fit with {len(free)} free parameters "
            f"(simplex={simplex}, strategy={strategy}, hesse={hesse}, "
            f"ncall={ncall}, tolerance={self.tolerance})"
        )
        result = self._run(
            free,
            names,
            fcn,
            grad,
            start_values=start_values,
            strategy=strategy,
            hesse=hesse,
            simplex=simplex,
            ncall=ncall,
        )
        self._log(f"single fit finished: {self._summary(result)}")
        return result

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

    def random_start(self, *, seed: int | None = None) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            parameter.name: self._draw_parameter(parameter, rng)
            for parameter in self.parameters
            if not parameter.fixed
        }

    def fit_multistart(
        self,
        n_starts: int = 20,
        *,
        seed: int | None = None,
        include_default: bool = False,
        simplex: bool = False,
        strategy: int = 1,
    ) -> MultiStartResult:
        if n_starts < 1:
            raise ValueError("n_starts must be at least 1")
        strategy = self._validate_strategy(strategy)

        free, names, fcn, grad = self._backend()
        rng = np.random.default_rng(seed)
        starts: list[dict[str, float]] = []
        if include_default:
            starts.append({parameter.name: float(parameter.value) for parameter in free})
        while len(starts) < n_starts:
            starts.append(
                {
                    parameter.name: self._draw_parameter(parameter, rng)
                    for parameter in free
                }
            )

        self._log(
            f"multistart with {len(free)} free parameters, {len(starts)} starts "
            f"(simplex={simplex}, strategy={strategy}, seed={seed})"
        )
        results = []
        for index, start in enumerate(starts, start=1):
            self._log(f"start {index}/{len(starts)} running")
            result = self._run(
                free,
                names,
                fcn,
                grad,
                start_values=start,
                strategy=strategy,
                hesse=False,
                simplex=simplex,
            )
            results.append(result)
            self._log(f"start {index}/{len(starts)} finished: {self._summary(result)}")
        results = tuple(results)

        valid_indices = tuple(
            index
            for index, result in enumerate(results)
            if bool(result.valid) and np.isfinite(float(result.fval))
        )
        if not valid_indices:
            raise RuntimeError("No valid finite minimum was found across the multistart scan")

        best_index = min(valid_indices, key=lambda index: float(results[index].fval))
        preliminary = results[best_index]
        self._log(
            f"selected start {best_index + 1}/{len(starts)} for strategy-2/HESSE "
            f"refinement: {self._summary(preliminary)}"
        )
        best_values = {name: float(preliminary.values[name]) for name in names}
        best = self._run(
            free,
            names,
            fcn,
            grad,
            start_values=best_values,
            strategy=2,
            hesse=True,
            simplex=False,
        )
        if not bool(best.valid) or not np.isfinite(float(best.fval)):
            raise RuntimeError("Best multistart solution is invalid after HESSE")
        self._log(f"final refined minimum: {self._summary(best)}")

        return MultiStartResult(best=best, results=results, starts=tuple(starts))