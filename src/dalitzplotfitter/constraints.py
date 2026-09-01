"""External likelihood constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jax import Array

Parameters = Mapping[str, object]


def _resolve(value: object, parameters: Parameters):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


@dataclass(frozen=True)
class GaussianConstraint:
    """Gaussian penalty ``0.5*((x-mu)/sigma)^2`` up to an additive constant."""

    parameter: object
    mean: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("GaussianConstraint sigma must be positive")

    def __call__(self, parameters: Parameters) -> Array:
        value = jnp.asarray(_resolve(self.parameter, parameters))
        return 0.5 * ((value - self.mean) / self.sigma) ** 2


@dataclass(frozen=True)
class ConstrainedNLL:
    """Add one or more external constraints to an existing NLL."""

    nll: object
    constraints: tuple[object, ...]

    def __init__(self, nll: object, *constraints: object):
        object.__setattr__(self, "nll", nll)
        object.__setattr__(self, "constraints", tuple(constraints))

    def __call__(self, parameters: Parameters) -> Array:
        total = jnp.asarray(self.nll(parameters))
        for constraint in self.constraints:
            total = total + jnp.asarray(constraint(parameters))
        return total


__all__ = ["ConstrainedNLL", "GaussianConstraint"]
