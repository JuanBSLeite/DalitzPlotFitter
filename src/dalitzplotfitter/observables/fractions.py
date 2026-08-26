"""Fit and interference fractions from a normalization matrix."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.integration import matrix_normalization


def fit_fractions(coefficients: Array, matrix: Array) -> Array:
    coefficients = jnp.asarray(coefficients)
    matrix = jnp.asarray(matrix)
    total = matrix_normalization(coefficients, matrix)
    diagonal = jnp.real(jnp.conj(coefficients) * jnp.diag(matrix) * coefficients)
    return diagonal / total


def interference_fractions(coefficients: Array, matrix: Array) -> Array:
    coefficients = jnp.asarray(coefficients)
    matrix = jnp.asarray(matrix)
    total = matrix_normalization(coefficients, matrix)
    pair = 2.0 * jnp.real(
        jnp.conj(coefficients)[:, None] * matrix * coefficients[None, :]
    ) / total
    upper = jnp.triu(pair, k=1)
    return upper + upper.T
