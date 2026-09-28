"""Pole lineshapes."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..context import ResonanceContext


@dataclass(frozen=True)
class Pole:
    """Simple fixed-width pole in m (Laura++ Eq. 37)."""

    def __call__(self, mass, context: ResonanceContext):
        m = jnp.asarray(mass)
        m0 = jnp.asarray(context.pole_mass)
        gamma0 = jnp.asarray(context.pole_width)
        return 1.0 / (m - m0 - 0.5j * gamma0)


@dataclass(frozen=True)
class SigmaPole:
    """The simple ``f0(500)`` pole used by the LHCb isobar model.

    The convention in the paper is ``sqrt(s_sigma) = m_sigma - i Gamma_sigma``
    and ``A_sigma(m) = 1 / (s_sigma - m**2)``.  In particular, the width is
    not divided by two here.
    """

    def __call__(self, mass, context: ResonanceContext):
        m = jnp.asarray(mass)
        pole = jnp.asarray(context.pole_mass) - 1j * jnp.asarray(context.pole_width)
        return 1.0 / (pole**2 - m**2)


__all__ = ["Pole", "SigmaPole"]
