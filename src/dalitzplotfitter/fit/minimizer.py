"""Minuit driver using JAX values and automatic gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .parameters import Parameter


class Minimizer:
    """Minimize a mapping-based JAX objective with iminuit."""

    def __init__(self, objective: Callable, parameters: Sequence[Parameter]):
        self.objective = objective
        self.parameters = tuple(parameters)

    def fit(self):
        from iminuit import Minuit

        free = tuple(parameter for parameter in self.parameters if not parameter.fixed)
        fixed = {parameter.name: parameter.value for parameter in self.parameters if parameter.fixed}
        names = tuple(parameter.name for parameter in free)
        start = tuple(parameter.value for parameter in free)

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
        for parameter in free:
            if parameter.bounds is not None:
                minuit.limits[parameter.name] = parameter.bounds
        minuit.migrad()
        minuit.hesse()
        return minuit
