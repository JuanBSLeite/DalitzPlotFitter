"""Cached normalization matrices for linear amplitude coefficients."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def normalization_matrix(components: Array, weights: Array, efficiency: Array | None = None) -> Array:
    components = jnp.asarray(components)
    weights = jnp.asarray(weights)
    if components.ndim != 2:
        raise ValueError("components must have shape (n_events, n_components)")
    if weights.shape != (components.shape[0],):
        raise ValueError("weights must have shape (n_events,)")
    if efficiency is None:
        efficiency = jnp.ones_like(weights)
    total_weights = weights * jnp.asarray(efficiency)
    return jnp.einsum(
        "n,ni,nj->ij",
        total_weights,
        jnp.conj(components),
        components,
    ) / components.shape[0]


def matrix_normalization(coefficients: Array, matrix: Array) -> Array:
    coefficients = jnp.asarray(coefficients)
    matrix = jnp.asarray(matrix)
    return jnp.real(jnp.conj(coefficients) @ matrix @ coefficients)
