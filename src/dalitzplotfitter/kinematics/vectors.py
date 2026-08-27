"""Minimal four-vector utilities for Laura++-style amplitudes."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def invariant_mass_squared(momentum: Array) -> Array:
    """Return Minkowski norm squared for ``(E, px, py, pz)`` four-vectors."""

    momentum = jnp.asarray(momentum)
    return momentum[..., 0] ** 2 - jnp.sum(momentum[..., 1:] ** 2, axis=-1)
