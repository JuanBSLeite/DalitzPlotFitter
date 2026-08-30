"""Uniform Monte Carlo sampling over the physical three-body Dalitz region."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .dalitz_grid import DalitzGrid, dalitz_s13_limits
from .sample import PhaseSpaceSample


@dataclass(frozen=True)
class DalitzMC:
    """Uniform Monte Carlo sample in physical ``(s12, s13)`` Dalitz area.

    Random points are drawn uniformly in the same equal-area auxiliary
    coordinates ``(u, v) in [0, 1]^2`` used by :class:`DalitzGrid`.  Since that
    mapping has constant Jacobian ``A_DP``, the resulting points are uniform in
    physical ``ds12 ds13`` area and every point carries the same integration
    weight ``A_DP`` under the package convention ``mean(weights * f)``.

    This is deliberately different from :class:`PhaseSpaceMC`, which samples a
    factorized phase-space proposal and therefore carries event-dependent
    importance weights.
    """

    mother_mass: float
    masses: tuple[float, float, float]
    boundary_resolution: int = 20001

    def __post_init__(self) -> None:
        if len(self.masses) != 3:
            raise ValueError("DalitzMC requires exactly three daughter masses")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")
        if self.boundary_resolution < 4:
            raise ValueError("boundary_resolution must be at least 4")

    def _mapping(self):
        # Reuse exactly the same cumulative-area mapping as the deterministic
        # grid so the two normalization methods differ only by deterministic
        # midpoint quadrature versus random uniform sampling.
        return DalitzGrid(
            self.mother_mass,
            self.masses,
            resolution=2,
            boundary_resolution=self.boundary_resolution,
        )._area_mapping()

    @property
    def area(self):
        return self._mapping()[2]

    def generate(self, size: int, *, seed: int | None = None) -> PhaseSpaceSample:
        """Generate ``size`` independent uniform physical Dalitz points."""

        if size <= 0:
            raise ValueError("size must be positive")

        support, cumulative, area = self._mapping()
        key = jax.random.PRNGKey(0 if seed is None else int(seed))
        key_u, key_v = jax.random.split(key)
        u = jax.random.uniform(key_u, (int(size),), dtype=support.dtype)
        v = jax.random.uniform(key_v, (int(size),), dtype=support.dtype)

        s12 = jnp.interp(u * area, cumulative, support)
        low, high = dalitz_s13_limits(
            s12,
            mother_mass=self.mother_mass,
            masses=self.masses,
        )
        s13 = low + (high - low) * v

        m1, m2, m3 = self.masses
        s23 = (
            self.mother_mass**2
            + m1**2
            + m2**2
            + m3**2
            - s12
            - s13
        )
        weights = jnp.full_like(s12, area)

        return PhaseSpaceSample(
            s12=s12,
            s13=s13,
            s23=s23,
            weights=weights,
        )
