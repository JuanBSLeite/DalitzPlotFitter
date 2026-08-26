"""JAX-native three-body phase-space sampling."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from .dalitz import s12_limits, s13_from_s12_s23, s23_limits
from .four_vectors import four_momenta_from_dalitz


@dataclass(frozen=True)
class PhaseSpaceSample:
    s12: Array
    s13: Array
    s23: Array
    weights: Array
    p1: Array | None = None
    p2: Array | None = None
    p3: Array | None = None

    @property
    def size(self) -> int:
        return int(self.s12.shape[0])

    def as_dict(self) -> dict[str, Array]:
        data = {"s12": self.s12, "s13": self.s13, "s23": self.s23}
        if self.p1 is not None and self.p2 is not None and self.p3 is not None:
            data.update({"p1": self.p1, "p2": self.p2, "p3": self.p3})
        return data

    def as_momentum_dict(self) -> dict[int, Array]:
        """Return momenta using AmpForm/TensorWaves final-state IDs 0, 1, 2."""

        if self.p1 is None or self.p2 is None or self.p3 is None:
            raise ValueError("This phase-space sample does not contain four-momenta")
        return {0: self.p1, 1: self.p2, 2: self.p3}


@dataclass(frozen=True)
class ThreeBodyPhaseSpace:
    mother_mass: float
    masses: tuple[float, float, float]

    def __post_init__(self) -> None:
        if min(self.masses) < 0.0:
            raise ValueError("Final-state masses must be non-negative")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")

    def generate(
        self,
        key: Array,
        size: int,
        *,
        with_momenta: bool = True,
    ) -> PhaseSpaceSample:
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

        if not with_momenta:
            return PhaseSpaceSample(s12=s12, s13=s13, s23=s23, weights=weights)

        p1, p2, p3 = four_momenta_from_dalitz(
            s12,
            s23,
            self.mother_mass,
            self.masses,
        )
        return PhaseSpaceSample(
            s12=s12,
            s13=s13,
            s23=s23,
            weights=weights,
            p1=p1,
            p2=p2,
            p3=p3,
        )
