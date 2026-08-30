"""Simple fixed-width pole lineshape."""

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


__all__ = ["Pole"]
