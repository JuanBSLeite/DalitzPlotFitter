"""Deterministic midpoint grid for three-body Dalitz integration."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .sample import PhaseSpaceSample


def _kallen(x, y, z):
    return x**2 + y**2 + z**2 - 2.0 * x * y - 2.0 * x * z - 2.0 * y * z


def dalitz_s13_limits(
    s12,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
):
    """Return the physical ``s13`` limits at fixed ``s12``.

    The limits are evaluated in the rest frame of the ``(1,2)`` pair. They are
    exact for a spinless three-body kinematic boundary and are independent of the
    decay dynamics.
    """

    s12 = jnp.asarray(s12)
    m1, m2, m3 = masses
    root_s12 = jnp.sqrt(s12)

    e1 = (s12 + m1**2 - m2**2) / (2.0 * root_s12)
    e3 = (mother_mass**2 - s12 - m3**2) / (2.0 * root_s12)
    q = jnp.sqrt(jnp.maximum(_kallen(s12, m1**2, m2**2), 0.0)) / (
        2.0 * root_s12
    )
    p = jnp.sqrt(
        jnp.maximum(_kallen(mother_mass**2, s12, m3**2), 0.0)
    ) / (2.0 * root_s12)

    common = m1**2 + m3**2 + 2.0 * e1 * e3
    spread = 2.0 * q * p
    return common - spread, common + spread


@dataclass(frozen=True)
class DalitzGrid:
    """Regular ``N x N`` midpoint grid clipped to the physical Dalitz region.

    The proposal is a Cartesian grid in ``(s12, s13)``. Only bin centres inside
    the physical boundary are retained. All retained bins have the same area, so
    there is no importance sampling and no event-dependent phase-space weight.

    ``PhaseSpaceSample.weights`` follows the package-wide mean-estimator
    convention. Each retained point therefore receives ``N_valid * cell_area``;
    consequently ``mean(weights * f)`` is exactly the midpoint quadrature
    ``cell_area * sum(f)``.
    """

    mother_mass: float
    masses: tuple[float, float, float]
    resolution: int = 800

    def __post_init__(self) -> None:
        if self.resolution < 2:
            raise ValueError("DalitzGrid resolution must be at least 2")
        if len(self.masses) != 3:
            raise ValueError("DalitzGrid requires exactly three daughter masses")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")

    def sample(self) -> PhaseSpaceSample:
        """Return physical midpoint centres with constant quadrature weight."""

        m1, m2, m3 = self.masses
        n = int(self.resolution)

        s12_min = (m1 + m2) ** 2
        s12_max = (self.mother_mass - m3) ** 2
        s13_min_global = (m1 + m3) ** 2
        s13_max_global = (self.mother_mass - m2) ** 2

        ds12 = (s12_max - s12_min) / n
        ds13 = (s13_max_global - s13_min_global) / n

        s12_centres = s12_min + (jnp.arange(n) + 0.5) * ds12
        s13_centres = s13_min_global + (jnp.arange(n) + 0.5) * ds13
        s12_mesh, s13_mesh = jnp.meshgrid(
            s12_centres,
            s13_centres,
            indexing="ij",
        )

        low, high = dalitz_s13_limits(
            s12_mesh,
            mother_mass=self.mother_mass,
            masses=self.masses,
        )
        physical = (s13_mesh >= low) & (s13_mesh <= high)

        s12 = s12_mesh[physical]
        s13 = s13_mesh[physical]
        s23 = (
            self.mother_mass**2
            + m1**2
            + m2**2
            + m3**2
            - s12
            - s13
        )

        n_valid = s12.shape[0]
        # Package integrators use mean(weights * f). For a midpoint grid the
        # desired quadrature is cell_area * sum(f), hence this constant weight.
        quadrature_weight = n_valid * ds12 * ds13
        weights = jnp.full_like(s12, quadrature_weight)

        return PhaseSpaceSample(
            s12=s12,
            s13=s13,
            s23=s23,
            weights=weights,
        )
