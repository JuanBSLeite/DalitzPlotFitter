"""Gounaris-Sakurai lineshape."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..context import ResonanceContext
from .common import breakup_momentum, energy_dependent_width


@dataclass(frozen=True)
class GounarisSakurai:
    """Laura++ Gounaris-Sakurai lineshape for rho -> pi pi."""

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 1:
            raise ValueError("GounarisSakurai is defined for spin-1 rho-like states")

        m = jnp.asarray(mass)
        m0 = jnp.asarray(context.pole_mass)
        gamma0 = jnp.asarray(context.pole_width)
        mpi = 0.5 * (context.daughter_masses[0] + context.daughter_masses[1])
        q = breakup_momentum(m, mpi, mpi)
        q0 = breakup_momentum(m0, mpi, mpi)

        def h(x, qx):
            safe_q = jnp.where(qx > 0.0, qx, jnp.finfo(jnp.asarray(x).dtype).tiny)
            return (2.0 / jnp.pi) * (safe_q / x) * jnp.log((x + 2.0 * safe_q) / (2.0 * mpi))

        h_m = h(m, q)
        h_0 = h(m0, q0)
        dh_dm2_0 = h_0 * (1.0 / (8.0 * q0**2) - 1.0 / (2.0 * m0**2)) + 1.0 / (2.0 * jnp.pi * m0**2)
        f_m = gamma0 * m0**2 / q0**3 * (
            q**2 * (h_m - h_0) + (m0**2 - m**2) * q0**2 * dh_dm2_0
        )
        d = (
            (3.0 / jnp.pi) * (mpi**2 / q0**2) * jnp.log((m0 + 2.0 * q0) / (2.0 * mpi))
            + m0 / (2.0 * jnp.pi * q0)
            - mpi**2 * m0 / (jnp.pi * q0**3)
        )
        width = energy_dependent_width(m, context)
        numerator = 1.0 + d * gamma0 / m0
        return numerator / (m0**2 - m**2 + f_m - 1j * m0 * width)


__all__ = ["GounarisSakurai"]
