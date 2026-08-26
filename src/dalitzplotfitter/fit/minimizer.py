"""Minuit driver using JAX values and automatic gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .parameters import Parameter


class Minimizer:
    """Minimize a mapping-based JAX objective with iminuit.

    The default ``errordef=0.5`` is the Minuit convention for a negative
    log-likelihood. It makes HESSE uncertainties correspond to Delta(NLL)=0.5.
    Use a different value only when minimizing an objective with another
    statistical convention.
    """

    def __init__(
        self,
        objective: Callable,
        parameters: Sequence[Parameter],
        *,
        errordef: float = 0.5,
    ):
        if errordef <= 0:
            raise ValueError("errordef must be positive")
        self.objective = objective
        self.parameters = tuple(parameters)
        self.errordef = float(errordef)

    def fit(self):
        from iminuit import Minuit

        free = tuple(parameter for parameter in self.parameters if not parameter.fixed)
        fixed = {parameter.name: parameter.value for parameter in self.parameters if parameter.fixed}
        names = tuple(parameter.name for parameter in free)
        start = tuple(parameter.value for parameter in free)

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

        minuit = Minuit(fcn, *start, name=names, grad=grad)
        minuit.errordef = self.errordef
        for parameter in free:
            if parameter.bounds is not None:
                minuit.limits[parameter.name] = parameter.bounds
            if parameter.step is not None:
                if parameter.step <= 0:
                    raise ValueError(
                        f"Parameter step must be positive for {parameter.name!r}"
                    )
                minuit.errors[parameter.name] = parameter.step
        minuit.migrad()
        minuit.hesse()
        return minuit
