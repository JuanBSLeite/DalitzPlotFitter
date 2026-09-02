"""Relativistic Breit-Wigner lineshape."""

from __future__ import annotations

from dataclasses import dataclass

from ..context import ResonanceContext
from .common import energy_dependent_width


@dataclass(frozen=True)
class RelativisticBreitWigner:
    """Relativistic Breit-Wigner lineshape with running width."""

    def __call__(self, mass, context: ResonanceContext):
        width = energy_dependent_width(mass, context)
        m0 = context.pole_mass
        return 1.0 / (m0**2 - mass**2 - 1j * m0 * width)


__all__ = ["RelativisticBreitWigner"]
