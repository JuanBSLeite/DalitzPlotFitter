"""Inverse-transform sampling for unweighted three-body Dalitz pseudo-data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.kinematics import PhaseSpaceSample


DensityFunction = Callable[[dict[str, object]], object]


def _kallen(x, y, z):
    return x**2 + y**2 + z**2 - 2.0 * x * y - 2.0 * x * z - 2.0 * y * z


def _s13_limits(
    s12: np.ndarray,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy version of the exact physical ``s13`` limits at fixed ``s12``."""

    s12 = np.asarray(s12, dtype=float)
    m1, m2, m3 = masses
    root_s12 = np.sqrt(s12)
    e1 = (s12 + m1**2 - m2**2) / (2.0 * root_s12)
    e3 = (mother_mass**2 - s12 - m3**2) / (2.0 * root_s12)
    q = np.sqrt(np.maximum(_kallen(s12, m1**2, m2**2), 0.0)) / (2.0 * root_s12)
    p = np.sqrt(np.maximum(_kallen(mother_mass**2, s12, m3**2), 0.0)) / (
        2.0 * root_s12
    )
    common = m1**2 + m3**2 + 2.0 * e1 * e3
    spread = 2.0 * q * p
    return common - spread, common + spread


def _cumulative_trapezoid(values: np.ndarray, coordinates: np.ndarray, *, axis: int) -> np.ndarray:
    """Small dependency-free cumulative trapezoidal integrator."""

    values = np.asarray(values, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    moved = np.moveaxis(values, axis, -1)
    if moved.shape[-1] != coordinates.size:
        raise ValueError("integration coordinate length does not match density axis")
    dx = np.diff(coordinates)
    increments = 0.5 * (moved[..., :-1] + moved[..., 1:]) * dx
    result = np.zeros_like(moved)
    result[..., 1:] = np.cumsum(increments, axis=-1)
    return np.moveaxis(result, -1, axis)


def _inverse_row(cdf: np.ndarray, coordinate: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    cdf = np.maximum.accumulate(np.asarray(cdf, dtype=float))
    coordinate = np.asarray(coordinate, dtype=float)
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0.0:
        return np.interp(quantiles, (0.0, 1.0), (coordinate[0], coordinate[-1]))
    cdf = cdf / cdf[-1]
    unique, indices = np.unique(cdf, return_index=True)
    points = coordinate[indices]
    if unique[0] > 0.0:
        unique = np.concatenate(([0.0], unique))
        points = np.concatenate(([coordinate[0]], points))
    if unique[-1] < 1.0:
        unique = np.concatenate((unique, [1.0]))
        points = np.concatenate((points, [coordinate[-1]]))
    return np.interp(quantiles, unique, points)


def _momenta_from_invariants(
    s12: np.ndarray,
    s13: np.ndarray,
    s23: np.ndarray,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct isotropically oriented parent-rest-frame four-momenta."""

    s12 = np.asarray(s12, dtype=float)
    s13 = np.asarray(s13, dtype=float)
    s23 = np.asarray(s23, dtype=float)
    size = s12.size
    m1, m2, m3 = masses
    mother2 = mother_mass**2

    e1 = (mother2 + m1**2 - s23) / (2.0 * mother_mass)
    e2 = (mother2 + m2**2 - s13) / (2.0 * mother_mass)
    e3 = (mother2 + m3**2 - s12) / (2.0 * mother_mass)
    p1_mag = np.sqrt(np.maximum(e1**2 - m1**2, 0.0))
    p2_mag = np.sqrt(np.maximum(e2**2 - m2**2, 0.0))

    pair_dot = 0.5 * (s12 - m1**2 - m2**2)
    spatial_dot = e1 * e2 - pair_dot
    denominator = p1_mag * p2_mag
    cos12 = np.divide(
        spatial_dot,
        denominator,
        out=np.ones_like(spatial_dot),
        where=denominator > 0.0,
    )
    cos12 = np.clip(cos12, -1.0, 1.0)
    sin12 = np.sqrt(np.maximum(1.0 - cos12**2, 0.0))

    cos_theta = 2.0 * rng.random(size) - 1.0
    phi = 2.0 * np.pi * rng.random(size)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    n1 = np.stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta), axis=1
    )

    reference = np.zeros_like(n1)
    use_x = np.abs(n1[:, 2]) > 0.9
    reference[:, 2] = 1.0
    reference[use_x] = np.asarray([1.0, 0.0, 0.0])
    e_perp1 = np.cross(reference, n1)
    norm = np.linalg.norm(e_perp1, axis=1)
    e_perp1 = e_perp1 / norm[:, None]
    e_perp2 = np.cross(n1, e_perp1)

    alpha = 2.0 * np.pi * rng.random(size)
    transverse = (
        np.cos(alpha)[:, None] * e_perp1 + np.sin(alpha)[:, None] * e_perp2
    )
    n2 = cos12[:, None] * n1 + sin12[:, None] * transverse

    spatial1 = p1_mag[:, None] * n1
    spatial2 = p2_mag[:, None] * n2
    spatial3 = -(spatial1 + spatial2)
    p1 = np.concatenate((e1[:, None], spatial1), axis=1)
    p2 = np.concatenate((e2[:, None], spatial2), axis=1)
    p3 = np.concatenate((e3[:, None], spatial3), axis=1)
    return p1, p2, p3


@dataclass(frozen=True)
class DalitzInverseTransformSampler:
    """Prepared interpolated Rosenblatt sampler on a physical Dalitz domain.

    The target density is understood with respect to the conventional
    ``ds12 ds13`` Dalitz measure. Preparation tabulates the density on a
    rectangular ``(m12, v)`` grid, where

    ``s13 = s13_min(s12) + v * (s13_max(s12) - s13_min(s12))``.

    The transformed marginal therefore includes the exact Jacobian
    ``2*m12 * (s13_max-s13_min)``. Conditional inverse CDFs are tabulated as
    quantiles and bilinearly interpolated during generation. Generation itself
    has no rejection and produces continuous, duplicate-free invariants.
    """

    mother_mass: float
    masses: tuple[float, float, float]
    m12_grid: np.ndarray
    marginal_cdf: np.ndarray
    marginal_m12: np.ndarray
    quantile_levels: np.ndarray
    conditional_quantiles: np.ndarray

    @classmethod
    def prepare(
        cls,
        mother_mass: float,
        masses: tuple[float, float, float],
        density_function: DensityFunction,
        *,
        resolution: int = 1024,
        quantile_resolution: int | None = None,
    ) -> "DalitzInverseTransformSampler":
        if resolution < 16:
            raise ValueError("inverse-transform resolution must be at least 16")
        if quantile_resolution is None:
            quantile_resolution = resolution
        if quantile_resolution < 16:
            raise ValueError("inverse-transform quantile resolution must be at least 16")
        if len(masses) != 3:
            raise ValueError("inverse-transform sampler requires exactly three daughter masses")
        if mother_mass <= sum(masses):
            raise ValueError("mother mass must exceed the three-body threshold")

        m1, m2, m3 = masses
        m_min = m1 + m2
        m_max = mother_mass - m3
        m12_grid = np.linspace(m_min, m_max, int(resolution), dtype=float)
        m12_eval = m12_grid.copy()
        m12_eval[0] = np.nextafter(m_min, m_max)
        m12_eval[-1] = np.nextafter(m_max, m_min)
        s12_rows = m12_eval**2
        low, high = _s13_limits(s12_rows, mother_mass=mother_mass, masses=masses)
        width = np.maximum(high - low, 0.0)

        v_grid = np.linspace(0.0, 1.0, int(resolution), dtype=float)
        s13 = low[:, None] + width[:, None] * v_grid[None, :]
        constant = mother_mass**2 + m1**2 + m2**2 + m3**2
        s12 = np.broadcast_to(s12_rows[:, None], s13.shape)
        s23 = constant - s12 - s13
        data = {
            "s12": jnp.asarray(s12.reshape(-1)),
            "s13": jnp.asarray(s13.reshape(-1)),
            "s23": jnp.asarray(s23.reshape(-1)),
        }
        try:
            density = np.asarray(
                jax.device_get(jnp.asarray(density_function(data))), dtype=float
            ).reshape(s13.shape)
        except KeyError as exc:
            raise ValueError(
                "inverse-transform toy generation supports densities expressed in "
                "Dalitz invariants s12/s13/s23; the supplied efficiency or veto "
                "requested another event field"
            ) from exc
        if density.shape != s13.shape:
            raise ValueError("inverse-transform density must return one value per grid point")
        if np.any(~np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("inverse-transform density must be finite and non-negative")

        conditional_cumulative = _cumulative_trapezoid(density, v_grid, axis=1)
        row_integral_v = conditional_cumulative[:, -1]
        marginal_density = 2.0 * m12_eval * width * row_integral_v
        marginal_cumulative = _cumulative_trapezoid(
            marginal_density, m12_grid, axis=0
        )
        total = float(marginal_cumulative[-1])
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("inverse-transform target density has zero or invalid integral")
        marginal_cdf_full = np.maximum.accumulate(marginal_cumulative / total)
        unique_cdf, unique_indices = np.unique(marginal_cdf_full, return_index=True)
        marginal_m12 = m12_grid[unique_indices]
        if unique_cdf[0] > 0.0:
            unique_cdf = np.concatenate(([0.0], unique_cdf))
            marginal_m12 = np.concatenate(([m12_grid[0]], marginal_m12))
        if unique_cdf[-1] < 1.0:
            unique_cdf = np.concatenate((unique_cdf, [1.0]))
            marginal_m12 = np.concatenate((marginal_m12, [m12_grid[-1]]))

        quantile_levels = np.linspace(0.0, 1.0, int(quantile_resolution), dtype=float)
        conditional_quantiles = np.empty(
            (m12_grid.size, quantile_levels.size), dtype=float
        )
        for row in range(m12_grid.size):
            conditional_quantiles[row] = _inverse_row(
                conditional_cumulative[row], v_grid, quantile_levels
            )

        return cls(
            mother_mass=float(mother_mass),
            masses=tuple(float(value) for value in masses),
            m12_grid=m12_grid,
            marginal_cdf=unique_cdf,
            marginal_m12=marginal_m12,
            quantile_levels=quantile_levels,
            conditional_quantiles=conditional_quantiles,
        )

    def generate(self, size: int, *, seed: int | None = None) -> PhaseSpaceSample:
        if size <= 0:
            raise ValueError("size must be positive")
        rng = np.random.default_rng(seed)
        u_marginal = rng.random(size)
        u_conditional = rng.random(size)
        m12 = np.interp(u_marginal, self.marginal_cdf, self.marginal_m12)

        row = np.searchsorted(self.m12_grid, m12, side="right") - 1
        row = np.clip(row, 0, self.m12_grid.size - 2)
        m0 = self.m12_grid[row]
        m1_grid = self.m12_grid[row + 1]
        row_fraction = np.divide(
            m12 - m0,
            m1_grid - m0,
            out=np.zeros_like(m12),
            where=m1_grid > m0,
        )

        q_position = u_conditional * (self.quantile_levels.size - 1)
        q_index = np.floor(q_position).astype(np.int64)
        q_index = np.clip(q_index, 0, self.quantile_levels.size - 2)
        q_fraction = q_position - q_index

        table = self.conditional_quantiles
        v00 = table[row, q_index]
        v01 = table[row, q_index + 1]
        v10 = table[row + 1, q_index]
        v11 = table[row + 1, q_index + 1]
        v0 = v00 + q_fraction * (v01 - v00)
        v1 = v10 + q_fraction * (v11 - v10)
        v = np.clip(v0 + row_fraction * (v1 - v0), 0.0, 1.0)

        s12 = m12**2
        low, high = _s13_limits(
            s12, mother_mass=self.mother_mass, masses=self.masses
        )
        s13 = low + v * (high - low)
        m1, m2, m3 = self.masses
        s23 = self.mother_mass**2 + m1**2 + m2**2 + m3**2 - s12 - s13
        p1, p2, p3 = _momenta_from_invariants(
            s12,
            s13,
            s23,
            mother_mass=self.mother_mass,
            masses=self.masses,
            rng=rng,
        )
        return PhaseSpaceSample(
            s12=jnp.asarray(s12),
            s13=jnp.asarray(s13),
            s23=jnp.asarray(s23),
            weights=jnp.ones((size,), dtype=jnp.asarray(s12).dtype),
            p1=jnp.asarray(p1),
            p2=jnp.asarray(p2),
            p3=jnp.asarray(p3),
        )


__all__ = ["DalitzInverseTransformSampler"]
