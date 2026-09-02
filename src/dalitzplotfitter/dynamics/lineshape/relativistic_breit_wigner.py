"""Relativistic Breit-Wigner lineshape."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..context import ResonanceContext
from .common import (
    blatt_weisskopf_denominator,
    blatt_weisskopf_from_denominator,
    breakup_momentum,
    effective_pole_mass,
    energy_dependent_width,
)


@dataclass(frozen=True)
class RelativisticBreitWigner:
    """Relativistic Breit-Wigner lineshape with running width."""

    def prepare_mass(self, mass, context: ResonanceContext):
        """Prepare event-local factors independent of the fitted pole mass."""

        mass = jnp.asarray(mass)
        m1, m2 = context.daughter_masses
        q = breakup_momentum(mass, m1, m2)
        barrier_denominator = blatt_weisskopf_denominator(
            q,
            int(context.spin),
            context.resonance_radius,
        )
        return mass, q, barrier_denominator

    def evaluate_prepared(self, mass, prepared, context: ResonanceContext):
        """Evaluate the RBW using event-local quantities prepared once."""

        prepared_mass, q, barrier_denominator = prepared
        # ``mass`` is kept in the protocol for consistency with other prepared
        # lineshapes.  The stored mass is the exact event coordinate used when
        # preparing q and the barrier denominator.
        del mass
        m = jnp.asarray(prepared_mass)
        m1, m2 = context.daughter_masses
        pole_mass_for_momenta = effective_pole_mass(context)
        q0 = breakup_momentum(pole_mass_for_momenta, m1, m2)
        x_res = blatt_weisskopf_from_denominator(
            q0,
            barrier_denominator,
            int(context.spin),
            context.resonance_radius,
        )
        safe_mass = jnp.where(m > 0.0, m, jnp.nan)
        safe_q0 = jnp.where(q0 > 0.0, q0, jnp.nan)
        width = (
            context.pole_width
            * (q / safe_q0) ** (2 * int(context.spin) + 1)
            * (context.pole_mass / safe_mass)
            * x_res**2
        )
        m0 = context.pole_mass
        return 1.0 / (m0**2 - m**2 - 1j * m0 * width)

    def __call__(self, mass, context: ResonanceContext):
        width = energy_dependent_width(mass, context)
        m0 = context.pole_mass
        return 1.0 / (m0**2 - mass**2 - 1j * m0 * width)


__all__ = ["RelativisticBreitWigner"]
