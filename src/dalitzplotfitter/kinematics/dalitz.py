"""Three-body Dalitz-plot kinematics."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

Scalar = float | Array


def kallen(x: Scalar, y: Scalar, z: Scalar) -> Array:
    x, y, z = jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)
    return x*x + y*y + z*z - 2.0*(x*y + x*z + y*z)


def invariant_sum(mother_mass: Scalar, masses: tuple[Scalar, Scalar, Scalar]) -> Array:
    m1, m2, m3 = (jnp.asarray(m) for m in masses)
    m0 = jnp.asarray(mother_mass)
    return m0*m0 + m1*m1 + m2*m2 + m3*m3


def s12_limits(mother_mass: Scalar, masses: tuple[Scalar, Scalar, Scalar]) -> tuple[Array, Array]:
    m1, m2, m3 = (jnp.asarray(m) for m in masses)
    m0 = jnp.asarray(mother_mass)
    return (m1 + m2)**2, (m0 - m3)**2


def s23_limits(s12: Scalar, mother_mass: Scalar, masses: tuple[Scalar, Scalar, Scalar]) -> tuple[Array, Array]:
    s12 = jnp.asarray(s12)
    m0 = jnp.asarray(mother_mass)
    m1, m2, m3 = (jnp.asarray(m) for m in masses)
    lam12 = jnp.maximum(kallen(s12, m1*m1, m2*m2), 0.0)
    lam03 = jnp.maximum(kallen(m0*m0, s12, m3*m3), 0.0)
    root = jnp.sqrt(lam12 * lam03)
    center = (s12 - m1*m1 + m2*m2) * (m0*m0 - s12 - m3*m3)
    denominator = 2.0 * s12
    common = m2*m2 + m3*m3 + center / denominator
    half_width = root / denominator
    return common - half_width, common + half_width


def s13_from_s12_s23(s12: Scalar, s23: Scalar, mother_mass: Scalar, masses: tuple[Scalar, Scalar, Scalar]) -> Array:
    return invariant_sum(mother_mass, masses) - jnp.asarray(s12) - jnp.asarray(s23)


def inside_dalitz(s12: Scalar, s23: Scalar, mother_mass: Scalar, masses: tuple[Scalar, Scalar, Scalar], *, atol: float = 1e-12) -> Array:
    s12, s23 = jnp.asarray(s12), jnp.asarray(s23)
    low12, high12 = s12_limits(mother_mass, masses)
    low23, high23 = s23_limits(s12, mother_mass, masses)
    return ((s12 >= low12-atol) & (s12 <= high12+atol) & (s23 >= low23-atol) & (s23 <= high23+atol))
