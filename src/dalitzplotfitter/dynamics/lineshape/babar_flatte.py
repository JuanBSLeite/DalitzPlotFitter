"""BaBar mass-coupling Flatte convention used for f0(980)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..context import ResonanceContext


@dataclass(frozen=True)
class BaBarFlatte:
    """Flatte form used in BaBar B± -> K± pi∓ pi± (hep-ex/0507004).

    The paper defines

    Gamma_pi(m) = g_pi * sqrt(m^2 - 4 m_pi^2)
    Gamma_K(m)  = g_K  * sqrt(m^2 - 4 m_K^2)

    with analytic continuation below threshold, and uses the usual
    ``1 / (m0^2 - m^2 - i m0 Gamma(m))`` denominator.
    """

    g_pi: float = 0.11
    g_k: float = 0.36
    pion_mass: float = 0.13957039
    kaon_mass: float = 0.493677

    def __post_init__(self) -> None:
        if self.g_pi < 0.0 or self.g_k < 0.0:
            raise ValueError("BaBarFlatte couplings must be non-negative")
        if self.pion_mass <= 0.0 or self.kaon_mass <= 0.0:
            raise ValueError("BaBarFlatte daughter masses must be positive")

    @staticmethod
    def _threshold_root(mass, daughter_mass):
        m = jnp.asarray(mass)
        argument = (m**2 - 4.0 * daughter_mass**2).astype(jnp.complex128)
        return jnp.sqrt(argument)

    def widths(self, mass):
        gamma_pi = self.g_pi * self._threshold_root(mass, self.pion_mass)
        gamma_k = self.g_k * self._threshold_root(mass, self.kaon_mass)
        return gamma_pi, gamma_k

    def __call__(self, mass, context: ResonanceContext):
        gamma_pi, gamma_k = self.widths(mass)
        m = jnp.asarray(mass)
        m0 = jnp.asarray(context.pole_mass)
        return 1.0 / (m0**2 - m**2 - 1j * m0 * (gamma_pi + gamma_k))


__all__ = ["BaBarFlatte"]
