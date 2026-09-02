"""Physical Dalitz-boundary helpers."""

from __future__ import annotations

import jax.numpy as jnp


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
