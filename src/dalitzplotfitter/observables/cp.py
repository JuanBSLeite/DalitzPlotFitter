"""CP-asymmetry observables."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def cp_asymmetry(
    particle_intensity: Array,
    antiparticle_intensity: Array,
    *,
    floor: float = 1e-300,
) -> Array:
    """Return (I_antiparticle - I_particle)/(I_antiparticle + I_particle)."""

    particle = jnp.asarray(particle_intensity)
    antiparticle = jnp.asarray(antiparticle_intensity)
    denominator = jnp.clip(antiparticle + particle, min=floor)
    return (antiparticle - particle) / denominator
