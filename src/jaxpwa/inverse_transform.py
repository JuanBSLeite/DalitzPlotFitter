"""Inverse-transform sampling for unweighted three-body Dalitz pseudo-data."""

from __future__ import annotations

import math
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp

from jaxpwa.kinematics import (
    PhaseSpaceSample,
    dalitz_s13_limits,
    square_dalitz_to_invariants,
)
from jaxpwa.kinematics.phase_space_mc import _momenta_from_invariants

DensityFunction = Callable[[dict[str, object]], object]


def _cumulative_trapezoid(values, coordinates, *, axis: int) -> jax.Array:
    """Small dependency-free cumulative trapezoidal integrator."""

    values = jnp.asarray(values)
    coordinates = jnp.asarray(coordinates)
    moved = jnp.moveaxis(values, axis, -1)
    if moved.shape[-1] != coordinates.size:
        raise ValueError("integration coordinate length does not match density axis")
    dx = jnp.diff(coordinates)
    increments = 0.5 * (moved[..., :-1] + moved[..., 1:]) * dx
    result = jnp.concatenate(
        (jnp.zeros_like(moved[..., :1]), jnp.cumsum(increments, axis=-1)), axis=-1
    )
    return jnp.moveaxis(result, -1, axis)


def _inverse_row(cdf, coordinate, quantiles) -> jax.Array:
    cdf = jax.lax.cummax(jnp.asarray(cdf), axis=0)
    coordinate = jnp.asarray(coordinate)
    quantiles = jnp.asarray(quantiles)
    total = cdf[-1]
    valid = jnp.isfinite(total) & (total > 0.0)
    cdf = cdf / jnp.where(valid, total, 1.0)
    # Search on the full CDF: a plateau represents a jump of the inverse,
    # not a segment to interpolate across a forbidden interval.
    index = jnp.searchsorted(cdf, quantiles, side="right") - 1
    # The upper endpoint ends at the first CDF value of one, before any
    # trailing zero-density interval.
    index = jnp.where(
        quantiles >= 1.0, jnp.searchsorted(cdf, 1.0, side="left") - 1, index
    )
    index = jnp.clip(index, 0, cdf.size - 2)
    delta = cdf[index + 1] - cdf[index]
    fraction = jnp.where(
        delta > 0, (quantiles - cdf[index]) / jnp.where(delta > 0, delta, 1.0), 0.0
    )
    inverse = coordinate[index] + jnp.clip(fraction, 0, 1) * (
        coordinate[index + 1] - coordinate[index]
    )
    fallback = jnp.interp(
        quantiles, jnp.asarray((0.0, 1.0)), coordinate[jnp.asarray((0, -1))]
    )
    return jnp.where(valid, inverse, fallback)


@jax.jit
def _rosenblatt_tables(
    density: jax.Array,
    v_grid: jax.Array,
    m12_grid: jax.Array,
    row_jacobian: jax.Array,
    quantile_levels: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build the marginal CDF and per-row conditional inverse CDFs at once.

    ``row_jacobian`` maps each row's ``v`` integral to the marginal density
    (``2*m12*(s13_max-s13_min)`` on the Dalitz plane, one on the square).
    Returns ``(valid density, total integral, marginal CDF, quantile table)``.
    """

    valid = jnp.all(jnp.isfinite(density) & (density >= 0.0))
    conditional_cumulative = _cumulative_trapezoid(density, v_grid, axis=1)
    marginal_density = row_jacobian * conditional_cumulative[:, -1]
    marginal_cumulative = _cumulative_trapezoid(marginal_density, m12_grid, axis=0)
    total = marginal_cumulative[-1]
    marginal_cdf = jax.lax.cummax(marginal_cumulative / total, axis=0)
    # One conditional inverse CDF per m12 row.
    conditional_quantiles = jax.vmap(_inverse_row, in_axes=(0, None, None))(
        conditional_cumulative, v_grid, quantile_levels
    )
    return valid, total, marginal_cdf, conditional_quantiles


@jax.jit
def _support_summary(density: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return (all finite and non-negative, number of positive values)."""

    return jnp.all(jnp.isfinite(density) & (density >= 0)), jnp.sum(density > 0)


@partial(
    jax.jit,
    static_argnames=("size", "mother_mass", "masses", "square_dalitz_pair"),
)
def _draw_invariants(
    key: jax.Array,
    marginal_cdf: jax.Array,
    marginal_m12: jax.Array,
    m12_grid: jax.Array,
    quantile_levels: jax.Array,
    conditional_quantiles: jax.Array,
    *,
    size: int,
    mother_mass: float,
    masses: tuple[float, float, float],
    square_dalitz_pair: tuple[int, int] | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Draw invariants through the bilinearly interpolated inverse CDF tables."""

    key_marginal, key_conditional = jax.random.split(key)
    dtype = m12_grid.dtype
    u_marginal = jax.random.uniform(key_marginal, (size,), dtype=dtype)
    u_conditional = jax.random.uniform(key_conditional, (size,), dtype=dtype)
    m12 = _inverse_row(marginal_cdf, marginal_m12, u_marginal)

    row = jnp.searchsorted(m12_grid, m12, side="right") - 1
    row = jnp.clip(row, 0, m12_grid.size - 2)
    m0 = m12_grid[row]
    m1_grid = m12_grid[row + 1]
    row_fraction = jnp.where(
        m1_grid > m0, (m12 - m0) / jnp.where(m1_grid > m0, m1_grid - m0, 1.0), 0.0
    )

    q_position = u_conditional * (quantile_levels.size - 1)
    q_index = jnp.floor(q_position).astype(jnp.int32)
    q_index = jnp.clip(q_index, 0, quantile_levels.size - 2)
    q_fraction = q_position - q_index

    table = conditional_quantiles
    v00 = table[row, q_index]
    v01 = table[row, q_index + 1]
    v10 = table[row + 1, q_index]
    v11 = table[row + 1, q_index + 1]
    v0 = v00 + q_fraction * (v01 - v00)
    v1 = v10 + q_fraction * (v11 - v10)
    v = jnp.clip(v0 + row_fraction * (v1 - v0), 0.0, 1.0)

    if square_dalitz_pair is not None:
        return square_dalitz_to_invariants(
            m12,
            v,
            mother_mass=mother_mass,
            masses=masses,
            pair=square_dalitz_pair,
        )
    s12 = m12**2
    low, high = dalitz_s13_limits(s12, mother_mass=mother_mass, masses=masses)
    s13 = low + v * (high - low)
    m1, m2, m3 = masses
    s23 = mother_mass**2 + m1**2 + m2**2 + m3**2 - s12 - s13
    return s12, s13, s23


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
    rechecks the target support and replaces candidates with zero density.
    The density within the allowed support remains a grid approximation.

    With ``square_dalitz_pair`` set, the callback instead supplies a density
    per ``dm' dtheta'``. The same CDF algorithm operates directly on the unit
    square for that ordered pair, then converts generated points to invariants.
    The historical ``m12_grid``/``marginal_m12`` fields hold m' in this mode.
    """

    mother_mass: float
    masses: tuple[float, float, float]
    m12_grid: jax.Array
    marginal_cdf: jax.Array
    marginal_m12: jax.Array
    quantile_levels: jax.Array
    conditional_quantiles: jax.Array
    density_function: DensityFunction = field(repr=False, compare=False)
    square_dalitz_pair: tuple[int, int] | None = None

    @classmethod
    def prepare(
        cls,
        mother_mass: float,
        masses: tuple[float, float, float],
        density_function: DensityFunction,
        *,
        resolution: int = 1024,
        quantile_resolution: int | None = None,
        square_dalitz_pair: tuple[int, int] | None = None,
    ) -> DalitzInverseTransformSampler:
        if resolution < 16:
            raise ValueError("inverse-transform resolution must be at least 16")
        if quantile_resolution is None:
            quantile_resolution = resolution
        if quantile_resolution < 16:
            raise ValueError(
                "inverse-transform quantile resolution must be at least 16"
            )
        if len(masses) != 3:
            raise ValueError(
                "inverse-transform sampler requires exactly three daughter masses"
            )
        if mother_mass <= sum(masses):
            raise ValueError("mother mass must exceed the three-body threshold")
        mother_mass = float(mother_mass)
        masses = tuple(float(value) for value in masses)
        resolution = int(resolution)

        v_grid = jnp.linspace(0.0, 1.0, resolution)
        if square_dalitz_pair is not None:
            # The same Rosenblatt tables can use (m', theta') directly. The
            # density callback then supplies density per square area, avoiding
            # h/J singularities at the physical Dalitz boundary.
            m12_grid = jnp.linspace(0.0, 1.0, resolution)
            mp, tp = jnp.meshgrid(m12_grid, v_grid, indexing="ij")
            # At exactly m'=0 or 1 the helicity angle is undefined. Evaluate
            # one-sided limits inside the domain while keeping CDF endpoints.
            inv = square_dalitz_to_invariants(
                jnp.clip(mp.ravel(), 1e-6, 1.0 - 1e-6),
                jnp.clip(tp.ravel(), 1e-6, 1.0 - 1e-6),
                mother_mass=mother_mass,
                masses=masses,
                pair=square_dalitz_pair,
            )
            data = dict(zip(("s12", "s13", "s23"), inv, strict=True))
        else:
            m1, m2, m3 = masses
            m_min = jnp.asarray(m1 + m2, dtype=v_grid.dtype)
            m_max = jnp.asarray(mother_mass - m3, dtype=v_grid.dtype)
            m12_grid = jnp.linspace(m_min, m_max, resolution)
            m12_eval = (
                m12_grid.at[0]
                .set(jnp.nextafter(m_min, m_max))
                .at[-1]
                .set(jnp.nextafter(m_max, m_min))
            )
            s12_rows = m12_eval**2
            low, high = dalitz_s13_limits(
                s12_rows, mother_mass=mother_mass, masses=masses
            )
            width = jnp.maximum(high - low, 0.0)

            s13 = low[:, None] + width[:, None] * v_grid[None, :]
            constant = mother_mass**2 + m1**2 + m2**2 + m3**2
            s12 = jnp.broadcast_to(s12_rows[:, None], s13.shape)
            s23 = constant - s12 - s13
            data = {
                "s12": s12.reshape(-1),
                "s13": s13.reshape(-1),
                "s23": s23.reshape(-1),
            }
        try:
            density = jnp.asarray(density_function(data), dtype=v_grid.dtype)
        except KeyError as exc:
            raise ValueError(
                "inverse-transform toy generation supports densities expressed in "
                "Dalitz invariants s12/s13/s23; the supplied efficiency or veto "
                "requested another event field"
            ) from exc
        if density.size != resolution * resolution:
            raise ValueError(
                "inverse-transform density must return one value per grid point"
            )
        density = density.reshape((resolution, resolution))
        row_jacobian = (
            jnp.ones_like(m12_grid)
            if square_dalitz_pair is not None
            else 2.0 * m12_eval * width
        )
        quantile_levels = jnp.linspace(0.0, 1.0, int(quantile_resolution))
        valid, total, marginal_cdf_full, conditional_quantiles = _rosenblatt_tables(
            density, v_grid, m12_grid, row_jacobian, quantile_levels
        )
        valid, total = jax.device_get((valid, total))
        if not bool(valid):
            raise ValueError(
                "inverse-transform density must be finite and non-negative"
            )
        if not math.isfinite(float(total)) or float(total) <= 0.0:
            raise ValueError(
                "inverse-transform target density has zero or invalid integral"
            )

        return cls(
            mother_mass=mother_mass,
            masses=masses,
            m12_grid=m12_grid,
            marginal_cdf=marginal_cdf_full,
            marginal_m12=m12_grid,
            quantile_levels=quantile_levels,
            conditional_quantiles=conditional_quantiles,
            density_function=density_function,
            square_dalitz_pair=square_dalitz_pair,
        )

    def generate(
        self,
        size: int,
        *,
        seed: int | None = None,
        include_momenta: bool = True,
    ) -> PhaseSpaceSample:
        if size <= 0:
            raise ValueError("size must be positive")
        if seed is None:
            seed = secrets.randbits(32)
        key = jax.random.key(int(seed) % (2**32))
        draw_key, momenta_key = jax.random.split(key)
        accepted = []
        remaining = size
        for batch_key in jax.random.split(draw_key, 100):
            candidate = self._draw(remaining, key=batch_key)
            density = jnp.asarray(self.density_function(candidate.as_dict()))
            if density.shape != (remaining,):
                raise ValueError(
                    "inverse-transform density must return one value per event"
                )
            # Only two scalars reach the host; the event arrays stay on device.
            valid, n_positive = jax.device_get(_support_summary(density))
            if not bool(valid):
                raise ValueError(
                    "inverse-transform density must be finite and non-negative"
                )
            n_positive = int(n_positive)
            if n_positive == remaining:
                accepted.append(candidate)
            elif n_positive:
                indices = jnp.flatnonzero(density > 0, size=n_positive)
                accepted.append(candidate.take(indices))
            remaining -= n_positive
            if remaining == 0:
                break
        else:
            raise RuntimeError(
                "inverse-transform support rejection exhausted 100 batches; "
                "increase resolution or use accept-reject"
            )
        s12, s13, s23 = (
            getattr(accepted[0], name)
            if len(accepted) == 1
            else jnp.concatenate([getattr(sample, name) for sample in accepted])
            for name in ("s12", "s13", "s23")
        )
        p1 = p2 = p3 = None
        if include_momenta:
            mother_mass = jnp.asarray(self.mother_mass, dtype=s12.dtype)
            p1, p2, p3 = _momenta_from_invariants(
                momenta_key,
                mother_mass,
                jnp.asarray(self.masses, dtype=s12.dtype),
                s12,
                s13,
                s23,
                size=size,
            )
        return PhaseSpaceSample(
            s12=s12,
            s13=s13,
            s23=s23,
            weights=jnp.ones((size,), dtype=s12.dtype),
            p1=p1,
            p2=p2,
            p3=p3,
        )

    def _draw(self, size: int, *, key: jax.Array) -> PhaseSpaceSample:
        s12, s13, s23 = _draw_invariants(
            key,
            self.marginal_cdf,
            self.marginal_m12,
            self.m12_grid,
            self.quantile_levels,
            self.conditional_quantiles,
            size=size,
            mother_mass=self.mother_mass,
            masses=self.masses,
            square_dalitz_pair=self.square_dalitz_pair,
        )
        return PhaseSpaceSample(
            s12=s12,
            s13=s13,
            s23=s23,
            weights=jnp.ones((size,), dtype=s12.dtype),
        )


__all__ = ["DalitzInverseTransformSampler"]
