"""Kinematics used by the Laura++ covariant angular formalism."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from .four_vectors import invariant_mass_squared


def spatial_magnitude(momentum: Array) -> Array:
    """Return ``|p|`` for four-vectors stored as ``(E, px, py, pz)``."""

    momentum = jnp.asarray(momentum)
    return jnp.sqrt(jnp.maximum(jnp.sum(momentum[..., 1:] ** 2, axis=-1), 0.0))


def boost_to_rest_frame(momentum: Array, frame_momentum: Array) -> Array:
    """Lorentz-boost ``momentum`` to the rest frame of ``frame_momentum``.

    Both arrays use the project convention ``(E, px, py, pz)``. The function is
    vectorized over any leading dimensions.
    """

    momentum = jnp.asarray(momentum)
    frame_momentum = jnp.asarray(frame_momentum)

    frame_energy = frame_momentum[..., 0]
    beta = frame_momentum[..., 1:] / frame_energy[..., None]
    beta2 = jnp.sum(beta * beta, axis=-1)
    gamma = 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta2, 0.0))

    energy = momentum[..., 0]
    spatial = momentum[..., 1:]
    beta_dot_p = jnp.sum(beta * spatial, axis=-1)

    safe_beta2 = jnp.where(beta2 > 0.0, beta2, 1.0)
    spatial_factor = ((gamma - 1.0) * beta_dot_p / safe_beta2) - gamma * energy
    boosted_spatial = spatial + spatial_factor[..., None] * beta
    boosted_energy = gamma * (energy - beta_dot_p)

    # For an already-rest frame beta=0, the algebra above is finite because of
    # safe_beta2, but explicitly restoring the original spatial vector avoids
    # any unnecessary 0/0 sensitivity in transformed graphs.
    boosted_spatial = jnp.where(
        (beta2 > 0.0)[..., None],
        boosted_spatial,
        spatial,
    )
    return jnp.concatenate((boosted_energy[..., None], boosted_spatial), axis=-1)


@dataclass(frozen=True)
class CovariantKinematics:
    """Event-wise inputs required by the Laura++ covariant spin factor."""

    resonance_mass: Array
    p_star: Array
    p: Array
    q: Array
    cos_theta: Array


def covariant_kinematics(
    daughter: Array,
    partner: Array,
    bachelor: Array,
) -> CovariantKinematics:
    """Compute Laura++ covariant-spin kinematics from daughter four-vectors.

    ``daughter`` and ``partner`` form the resonance. ``daughter`` is the chosen
    resonance daughter that defines the helicity-angle sign convention;
    ``bachelor`` is the third final-state particle.

    The parent four-vector is reconstructed as ``daughter + partner + bachelor``.
    ``cos_theta`` is the cosine of the angle between the chosen daughter and the
    bachelor after both are boosted to the resonance rest frame. Exchanging the
    two equal-mass resonance daughters therefore reverses the sign of the odd-L
    angular term, as expected.
    """

    daughter = jnp.asarray(daughter)
    partner = jnp.asarray(partner)
    bachelor = jnp.asarray(bachelor)

    resonance = daughter + partner
    parent = resonance + bachelor

    resonance_mass2 = invariant_mass_squared(resonance)
    resonance_mass = jnp.sqrt(jnp.maximum(resonance_mass2, 0.0))

    bachelor_parent = boost_to_rest_frame(bachelor, parent)
    daughter_resonance = boost_to_rest_frame(daughter, resonance)
    bachelor_resonance = boost_to_rest_frame(bachelor, resonance)

    p_star = spatial_magnitude(bachelor_parent)
    q = spatial_magnitude(daughter_resonance)
    p = spatial_magnitude(bachelor_resonance)

    daughter_vec = daughter_resonance[..., 1:]
    bachelor_vec = bachelor_resonance[..., 1:]
    denominator = q * p
    safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
    cos_theta = jnp.sum(daughter_vec * bachelor_vec, axis=-1) / safe_denominator
    cos_theta = jnp.where(denominator > 0.0, cos_theta, 0.0)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)

    return CovariantKinematics(
        resonance_mass=resonance_mass,
        p_star=p_star,
        p=p,
        q=q,
        cos_theta=cos_theta,
    )
