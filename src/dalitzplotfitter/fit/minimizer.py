"""Minuit driver using JAX values and automatic gradients."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .parameters import Parameter


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
    ``tolerance`` is passed to ``Minuit.tol`` and therefore controls the EDM
    convergence target explicitly. The default is intentionally tighter than
    iminuit's generic default because amplitude fits can contain strongly
    correlated shape and complex-coefficient parameters.
    """

    def __init__(
        self,
        objective: Callable,
        parameters: Sequence[Parameter],
        *,
        errordef: float = 0.5,
        tolerance: float = 1e-10,
    ):
        if errordef <= 0:
            raise ValueError("errordef must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.objective = objective
        self.parameters = tuple(parameters)
        self.errordef = float(errordef)
        self.tolerance = float(tolerance)

    def _backend(self):
        free = tuple(parameter for parameter in self.parameters if not parameter.fixed)
        fixed = {
            parameter.name: parameter.value
            for parameter in self.parameters
            if parameter.fixed
        }
        names = tuple(parameter.name for parameter in free)
        if not free:
            raise ValueError("At least one free parameter is required")

        def vector_objective(vector):
            mapping = dict(fixed)
            mapping.update({name: vector[i] for i, name in enumerate(names)})
            return self.objective(mapping)

        value_and_grad = jax.jit(jax.value_and_grad(vector_objective))

        def fcn(*values):
            value, _ = value_and_grad(jnp.asarray(values))
            return float(value)

        def grad(*values):
            _, gradient = value_and_grad(jnp.asarray(values))
            return np.asarray(gradient, dtype=float)

        return free, names, fcn, grad

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
                    f"starting value {value} is below the lower bound {low} for {parameter.name!r}"
                )
            if high is not None and value > high:
                raise ValueError(
                    f"starting value {value} is above the upper bound {high} for {parameter.name!r}"
                )
        return values

    def _run(
        self,
        free,
        names,
        fcn,
        grad,
        *,
        start_values: Mapping[str, float] | None,
        run_hesse: bool,
        simplex: bool,
    ):
        from iminuit import Minuit

        supplied = self._validate_start_values(start_values)
        start = tuple(
            float(supplied.get(parameter.name, parameter.value))
            for parameter in free
        )
        minuit = Minuit(fcn, *start, name=names, grad=grad)
        minuit.errordef = self.errordef
        minuit.tol = self.tolerance
        for parameter in free:
            if parameter.bounds is not None:
                minuit.limits[parameter.name] = parameter.bounds
            if parameter.step is not None:
                minuit.errors[parameter.name] = parameter.step
        if simplex:
            minuit.simplex()
        minuit.migrad()
        if run_hesse:
            minuit.hesse()
        return minuit

    def fit(
        self,
        start_values: Mapping[str, float] | None = None,
        *,
        simplex: bool = False,
    ):
        """Run one MIGRAD/HESSE minimization from one starting point."""

        free, names, fcn, grad = self._backend()
        return self._run(
            free,
            names,
            fcn,
            grad,
            start_values=start_values,
            run_hesse=True,
            simplex=simplex,
        )

    @staticmethod
    def _draw_parameter(
        parameter: Parameter,
        rng: np.random.Generator,
    ) -> float:
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

    def random_start(
        self,
        *,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Draw one reproducible start for all free parameters."""

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
    ) -> MultiStartResult:
        """Run independent starts and select the valid solution with lowest NLL.

        All starts share a single compiled JAX value/gradient backend. MIGRAD is
        run for every start; HESSE is run only after selecting the lowest finite,
        valid minimum. No truth information is used in the selection.
        """

        if n_starts < 1:
            raise ValueError("n_starts must be at least 1")

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

        results = tuple(
            self._run(
                free,
                names,
                fcn,
                grad,
                start_values=start,
                run_hesse=False,
                simplex=simplex,
            )
            for start in starts
        )
        valid = tuple(
            result
            for result in results
            if bool(result.valid) and np.isfinite(float(result.fval))
        )
        if not valid:
            raise RuntimeError(
                "No valid finite minimum was found across the multistart scan"
            )

        preliminary = min(valid, key=lambda result: float(result.fval))
        best_values = {name: float(preliminary.values[name]) for name in names}
        best = self._run(
            free,
            names,
            fcn,
            grad,
            start_values=best_values,
            run_hesse=True,
            simplex=False,
        )
        if not bool(best.valid) or not np.isfinite(float(best.fval)):
            raise RuntimeError("Best multistart solution is invalid after HESSE")

        return MultiStartResult(
            best=best,
            results=results,
            starts=tuple(starts),
        )
