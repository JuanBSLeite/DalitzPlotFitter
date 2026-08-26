"""JAX-native three-body phase-space sampling."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from .dalitz import s12_limits, s13_from_s12_s23, s23_limits


@dataclass(frozen=True)
class PhaseSpaceSample:
    s12: Array
    s13: Array
    s23: Array
    weights: Array

    @property
    def size(self) -> int:
        return int(self.s12.shape[0])

    def as_dict(self) -> dict[str, Array]:
        return {"s12": self.s12, "s13": self.s13, "s23": self.s23}


@dataclass(frozen=True)
class ThreeBodyPhaseSpace:
    mother_mass: float
    masses: tuple[float, float, float]

    def __post_init__(self) -> None:
        if min(self.masses) < 0.0:
            raise ValueError("Final-state masses must be non-negative")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")

    def generate(self, key: Array, size: int) -> PhaseSpaceSample:
        if size <= 0:
            raise ValueError("size must be positive")
        u = jax.random.uniform(key, shape=(size, 2), minval=0.0, maxval=1.0)
        low12, high12 = s12_limits(self.mother_mass, self.masses)
        width12 = high12 - low12
        s12 = low12 + u[:, 0] * width12
        low23, high23 = s23_limits(s12, self.mother_mass, self.masses)
        width23 = high23 - low23
        s23 = low23 + u[:, 1] * width23
        s13 = s13_from_s12_s23(s12, s23, self.mother_mass, self.masses)
        weights = jnp.asarray(width12) * width23
        return PhaseSpaceSample(s12=s12, s13=s13, s23=s23, weights=weights)
