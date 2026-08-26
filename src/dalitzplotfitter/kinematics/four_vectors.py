"""Four-momentum reconstruction for three-body Dalitz samples."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .dalitz import kallen, s23_limits


def four_momenta_from_dalitz(
    s12: Array,
    s23: Array,
    mother_mass: float,
    masses: tuple[float, float, float],
) -> tuple[Array, Array, Array]:
    """Reconstruct one valid mother-rest-frame momentum configuration.

    The global event orientation is fixed conventionally. For an unpolarized scalar
    mother this rotation is physically irrelevant, while all invariant masses and
    helicity angles required by a three-body amplitude are preserved.

    The returned arrays follow the ``(E, px, py, pz)`` convention used by AmpForm
    and TensorWaves. Internal particles 1, 2, 3 map to TensorWaves final-state IDs
    0, 1, 2 respectively.
    """

    s12 = jnp.asarray(s12)
    s23 = jnp.asarray(s23)
    m0 = jnp.asarray(mother_mass)
    m1, m2, m3 = (jnp.asarray(mass) for mass in masses)

    sqrt_s12 = jnp.sqrt(jnp.maximum(s12, 0.0))
    pair_energy = (m0 * m0 + s12 - m3 * m3) / (2.0 * m0)
    pair_momentum = jnp.sqrt(
        jnp.maximum(kallen(m0 * m0, s12, m3 * m3), 0.0)
    ) / (2.0 * m0)

    beta = pair_momentum / pair_energy
    gamma = pair_energy / sqrt_s12

    energy1_star = (s12 + m1 * m1 - m2 * m2) / (2.0 * sqrt_s12)
    energy2_star = (s12 + m2 * m2 - m1 * m1) / (2.0 * sqrt_s12)
    breakup_momentum = jnp.sqrt(
        jnp.maximum(kallen(s12, m1 * m1, m2 * m2), 0.0)
    ) / (2.0 * sqrt_s12)

    low23, high23 = s23_limits(s12, mother_mass, masses)
    center23 = 0.5 * (low23 + high23)
    half_width23 = 0.5 * (high23 - low23)
    safe_half_width = jnp.where(half_width23 > 0.0, half_width23, 1.0)
    cos_theta2 = jnp.clip((s23 - center23) / safe_half_width, -1.0, 1.0)
    sin_theta2 = jnp.sqrt(jnp.maximum(1.0 - cos_theta2 * cos_theta2, 0.0))

    px2_star = breakup_momentum * sin_theta2
    pz2_star = breakup_momentum * cos_theta2
    px1_star = -px2_star
    pz1_star = -pz2_star

    energy1 = gamma * (energy1_star + beta * pz1_star)
    pz1 = gamma * (pz1_star + beta * energy1_star)
    energy2 = gamma * (energy2_star + beta * pz2_star)
    pz2 = gamma * (pz2_star + beta * energy2_star)

    zeros = jnp.zeros_like(s12)
    energy3 = (m0 * m0 - s12 + m3 * m3) / (2.0 * m0)

    p1 = jnp.stack((energy1, px1_star, zeros, pz1), axis=-1)
    p2 = jnp.stack((energy2, px2_star, zeros, pz2), axis=-1)
    p3 = jnp.stack((energy3, zeros, zeros, -pair_momentum), axis=-1)
    return p1, p2, p3


def invariant_mass_squared(momentum: Array) -> Array:
    """Return the invariant mass squared of one or more four-vectors."""

    momentum = jnp.asarray(momentum)
    return momentum[..., 0] ** 2 - jnp.sum(momentum[..., 1:] ** 2, axis=-1)
