"""Pole and LASS lineshapes.

The LASS implementation follows the form collected in Appendix A of Laura++.
The Pole model represents a scalar resonance directly through a complex pole in
``s`` with ``sqrt(s_pole) = m0 - i Gamma0/2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .context import ResonanceContext
from .lineshapes import breakup_momentum, effective_pole_mass


@dataclass(frozen=True)
class Pole:
    """Simple complex-pole lineshape in the invariant mass squared.

    The pole is parameterized as

    ``sqrt(s_pole) = m0 - i Gamma0/2``

    and the mass term is

    ``R(m) = 1 / (m^2 - s_pole)``.

    This is useful for very broad scalar states where a conventional running-
    width Breit-Wigner is not the desired parameterization.
    """

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("Pole is intended for scalar (spin-0) states")
        m = jnp.asarray(mass)
        pole_mass = jnp.asarray(context.pole_mass)
        pole_width = jnp.asarray(context.pole_width)
        sqrt_s_pole = pole_mass - 0.5j * pole_width
        s_pole = sqrt_s_pole**2
        return 1.0 / (m**2 - s_pole)


@dataclass(frozen=True)
class LASS:
    """Laura++-style elastic K-pi S-wave (effective range + K0*(1430)).

    The full amplitude is the coherent sum

    ``R = R_NR + exp(2 i delta_B) R_BW``

    with

    ``cot(delta_B) = 1/(a q) + r q / 2``.

    ``mode`` can be ``"full"``, ``"resonant"`` or ``"nonresonant"``, matching
    the Laura++ LASS, LASS_BW and LASS_NR choices.  ``cutoff`` suppresses only
    the effective-range nonresonant term above the requested invariant mass.
    """

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
            raise ValueError(
                "LASS mode must be 'full', 'resonant', or 'nonresonant'"
            )

    def terms(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("LASS is defined for a spin-0 K-pi S-wave")

        m = jnp.asarray(mass)
        m1, m2 = context.daughter_masses
        q = breakup_momentum(m, m1, m2)
        m0_for_q = effective_pole_mass(context)
        q0 = breakup_momentum(m0_for_q, m1, m2)
        m0 = jnp.asarray(context.pole_mass)
        gamma0 = jnp.asarray(context.pole_width)

        tiny = jnp.finfo(m.dtype).tiny
        safe_q = jnp.where(q > 0.0, q, tiny)
        safe_q0 = jnp.where(q0 > 0.0, q0, tiny)
        safe_m = jnp.where(m > 0.0, m, tiny)

        cot_delta_b = (
            1.0 / (self.scattering_length * safe_q)
            + 0.5 * self.effective_range * safe_q
        )
        elastic = 1.0 / (cot_delta_b - 1j)
        phase2 = (cot_delta_b + 1j) / (cot_delta_b - 1j)

        nonresonant = (m / safe_q) * elastic
        if self.cutoff is not None:
            nonresonant = jnp.where(m <= self.cutoff, nonresonant, 0.0j)

        numerator = m0 * gamma0 * (m0 / safe_q0)
        denominator = (
            m0**2
            - m**2
            - 1j * m0 * gamma0 * (q / safe_m) * (m0 / safe_q0)
        )
        resonant = phase2 * numerator / denominator
        return nonresonant, resonant

    def __call__(self, mass, context: ResonanceContext):
        nonresonant, resonant = self.terms(mass, context)
        if self.mode == "nonresonant":
            return nonresonant
        if self.mode == "resonant":
            return resonant
        return nonresonant + resonant
