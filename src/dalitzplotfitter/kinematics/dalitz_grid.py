"""Deterministic equal-area grid for three-body Dalitz integration."""

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
    """Return the exact physical ``s13`` limits at fixed ``s12``."""

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


def _dalitz_width(
    s12,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
):
    low, high = dalitz_s13_limits(
        s12,
        mother_mass=mother_mass,
        masses=masses,
    )
    return jnp.maximum(high - low, 0.0)


@dataclass(frozen=True)
class DalitzGrid:
    """Equal-area ``N x N`` grid mapped directly into the physical Dalitz region.

    A regular midpoint grid in auxiliary coordinates ``(u, v) in [0,1]^2`` is
    mapped into ``(s12, s13)``.  The first coordinate is defined by cumulative
    physical Dalitz area,

    ``u(s12) = integral[W(s) ds] / A_DP``,

    where ``W(s12) = s13_max(s12) - s13_min(s12)``.  The second coordinate is
    linear across the physical ``s13`` interval at fixed ``s12``.  This mapping
    has constant Jacobian ``A_DP``; consequently all ``N^2`` points are physical
    and have identical quadrature weight.

    ``PhaseSpaceSample.weights`` follows the package-wide estimator convention
    ``mean(weights * f)``.  Every point therefore stores the same value
    ``A_DP``, so the estimator is ``A_DP * mean(f)``.
    """

    mother_mass: float
    masses: tuple[float, float, float]
    resolution: int = 800
    boundary_resolution: int | None = None

    def __post_init__(self) -> None:
        if self.resolution < 2:
            raise ValueError("DalitzGrid resolution must be at least 2")
        if len(self.masses) != 3:
            raise ValueError("DalitzGrid requires exactly three daughter masses")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")
        if self.boundary_resolution is not None and self.boundary_resolution < 4:
            raise ValueError("boundary_resolution must be at least 4")

    def _area_mapping(self):
        """Return dense ``s12`` support, cumulative area and total Dalitz area."""

        m1, m2, m3 = self.masses
        s12_min = (m1 + m2) ** 2
        s12_max = (self.mother_mass - m3) ** 2

        # The dense one-dimensional support is used only to tabulate and invert
        # the cumulative physical area.  It is deterministic and independent of
        # the N x N integration grid itself.
        n_boundary = (
            int(self.boundary_resolution)
            if self.boundary_resolution is not None
            else max(4097, 8 * int(self.resolution) + 1)
        )
        support = jnp.linspace(s12_min, s12_max, n_boundary)
        width = _dalitz_width(
            support,
            mother_mass=self.mother_mass,
            masses=self.masses,
        )
        ds = support[1:] - support[:-1]
        increments = 0.5 * (width[:-1] + width[1:]) * ds
        cumulative = jnp.concatenate(
            (jnp.zeros((1,), dtype=support.dtype), jnp.cumsum(increments))
        )
        area = cumulative[-1]
        return support, cumulative, area

    @property
    def area(self):
        """Deterministic numerical area of the physical Dalitz region."""

        return self._area_mapping()[2]

    def sample(self) -> PhaseSpaceSample:
        """Return exactly ``resolution**2`` equal-area physical midpoint points."""

        m1, m2, m3 = self.masses
        n = int(self.resolution)
        support, cumulative, area = self._area_mapping()

        # Midpoints of equal-area strips in u.  Inverting the cumulative area
        # makes every s12 strip contain A_DP/N physical area.
        u = (jnp.arange(n, dtype=support.dtype) + 0.5) / n
        target_area = u * area
        s12_strip = jnp.interp(target_area, cumulative, support)

        low, high = dalitz_s13_limits(
            s12_strip,
            mother_mass=self.mother_mass,
            masses=self.masses,
        )
        width = high - low

        # Midpoints of N equal subdivisions across each physical s13 interval.
        v = (jnp.arange(n, dtype=support.dtype) + 0.5) / n
        s12 = jnp.repeat(s12_strip, n)
        s13 = (low[:, None] + width[:, None] * v[None, :]).reshape(-1)
        s23 = (
            self.mother_mass**2
            + m1**2
            + m2**2
            + m3**2
            - s12
            - s13
        )

        # Under the package convention mean(weights * f), storing A_DP for every
        # point gives A_DP * mean(f), the equal-area midpoint quadrature.
        weights = jnp.full_like(s12, area)

        return PhaseSpaceSample(
            s12=s12,
            s13=s13,
            s23=s23,
            weights=weights,
        )
