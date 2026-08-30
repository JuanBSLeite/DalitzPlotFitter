"""LASS K-pi S-wave lineshape."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..context import ResonanceContext
from .common import breakup_momentum, effective_pole_mass


@dataclass(frozen=True)
class LASS:
    """Laura++ K-pi S-wave: effective range plus K0*(1430)."""

    scattering_length: float = 2.07
    effective_range: float = 3.32
    cutoff: float | None = 1.8
    mode: str = "full"

    def __post_init__(self) -> None:
        if self.scattering_length <= 0.0:
            raise ValueError("LASS scattering_length must be positive")
        if self.effective_range < 0.0:
            raise ValueError("LASS effective_range must be non-negative")
        if self.cutoff is not None and self.cutoff <= 0.0:
            raise ValueError("LASS cutoff must be positive or None")
        if self.mode not in {"full", "resonant", "nonresonant"}:
            raise ValueError("LASS mode must be 'full', 'resonant', or 'nonresonant'")

    def terms(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("LASS is defined for a spin-0 K-pi S-wave")
        m = jnp.asarray(mass)
        m1, m2 = context.daughter_masses
        q = breakup_momentum(m, m1, m2)
        q0 = breakup_momentum(effective_pole_mass(context), m1, m2)
        m0 = jnp.asarray(context.pole_mass)
        gamma0 = jnp.asarray(context.pole_width)
        tiny = jnp.finfo(m.dtype).tiny
        safe_q = jnp.where(q > 0.0, q, tiny)
        safe_q0 = jnp.where(q0 > 0.0, q0, tiny)
        safe_m = jnp.where(m > 0.0, m, tiny)
        cot_delta_b = 1.0 / (self.scattering_length * safe_q) + 0.5 * self.effective_range * safe_q
        nonresonant = m / (safe_q * (cot_delta_b - 1j))
        if self.cutoff is not None:
            nonresonant = jnp.where(m <= self.cutoff, nonresonant, 0.0j)
        phase2 = (cot_delta_b + 1j) / (cot_delta_b - 1j)
        numerator = m0 * gamma0 * (m0 / safe_q0)
        denominator = m0**2 - m**2 - 1j * m0 * gamma0 * (q / safe_m) * (m0 / safe_q0)
        resonant = phase2 * numerator / denominator
        return nonresonant, resonant

    def __call__(self, mass, context: ResonanceContext):
        nonresonant, resonant = self.terms(mass, context)
        if self.mode == "nonresonant":
            return nonresonant
        if self.mode == "resonant":
            return resonant
        return nonresonant + resonant


__all__ = ["LASS"]
