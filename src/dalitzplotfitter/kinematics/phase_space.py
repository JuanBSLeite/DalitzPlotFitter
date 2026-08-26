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

    def as_momentum_dict(self) -> dict[str, Array]:
        """Return momenta with the labels expected by TensorWaves: p0, p1, p2."""

        if self.p1 is None or self.p2 is None or self.p3 is None:
            raise ValueError("This phase-space sample does not contain four-momenta")
        return {"p0": self.p1, "p1": self.p2, "p2": self.p3}

    def take(self, indices: Array) -> "PhaseSpaceSample":
        """Select events while preserving all available phase-space fields."""

        indices = jnp.asarray(indices)
        return PhaseSpaceSample(
            s12=self.s12[indices],
            s13=self.s13[indices],
            s23=self.s23[indices],
            weights=self.weights[indices],
            p1=None if self.p1 is None else self.p1[indices],
            p2=None if self.p2 is None else self.p2[indices],
            p3=None if self.p3 is None else self.p3[indices],
        )


@dataclass(frozen=True)
class ThreeBodyPhaseSpace:
    mother_mass: float
    masses: tuple[float, float, float]

    def __post_init__(self) -> None:
        if min(self.masses) < 0.0:
            raise ValueError("Final-state masses must be non-negative")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")

    @classmethod
    def from_reaction(cls, reaction: object) -> ThreeBodyPhaseSpace:
        """Build phase space directly from a three-body QRules reaction."""

        final_state = reaction.final_state
        if set(final_state) != {0, 1, 2}:
            raise ValueError("Expected exactly three final-state IDs: 0, 1, 2")
        initial_particle = reaction.initial_state[-1]
        masses = tuple(float(final_state[i].mass) for i in range(3))
        return cls(float(initial_particle.mass), masses)

    def from_unit_square(
        self,
        unit_points: Array,
        *,
        with_momenta: bool = True,
    ) -> PhaseSpaceSample:
        """Map deterministic points from ``[0, 1]^2`` onto the Dalitz plot.

        The first coordinate is mapped linearly to ``s12``. The second is mapped
        linearly between the kinematic ``s23`` limits at that ``s12``. The
        returned ``weights`` are the Jacobian of this transformation,

        ``ds12 ds23 = weights du1 du2``.

        This method is also used by deterministic envelope searches for toy
        accept-reject generation, so the random generator and the maximum search
        share exactly the same phase-space parametrization.
        """

        unit_points = jnp.asarray(unit_points)
        if unit_points.ndim != 2 or unit_points.shape[1] != 2:
            raise ValueError("unit_points must have shape (N, 2)")

        u1 = unit_points[:, 0]
        u2 = unit_points[:, 1]
        low12, high12 = s12_limits(self.mother_mass, self.masses)
        width12 = high12 - low12
        s12 = low12 + u1 * width12
        low23, high23 = s23_limits(s12, self.mother_mass, self.masses)
        width23 = high23 - low23
        s23 = low23 + u2 * width23
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

    def generate(
        self,
        key: Array,
        size: int,
        *,
        with_momenta: bool = True,
    ) -> PhaseSpaceSample:
        if size <= 0:
            raise ValueError("size must be positive")
        unit_points = jax.random.uniform(
            key,
            shape=(size, 2),
            minval=0.0,
            maxval=1.0,
        )
        return self.from_unit_square(unit_points, with_momenta=with_momenta)
