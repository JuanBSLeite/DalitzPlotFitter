"""Numerical resonance dynamics."""

from .angular import CovariantAngular, covariant_spin_factor
from .context import ResonanceContext
from .lineshapes import (
    RelativisticBreitWigner,
    blatt_weisskopf_from_momenta,
    breakup_momentum,
    energy_dependent_width,
    kallen,
)
from .resonance import ResonanceAmplitude

__all__ = [
    "CovariantAngular",
    "RelativisticBreitWigner",
    "ResonanceAmplitude",
    "ResonanceContext",
    "blatt_weisskopf_from_momenta",
    "breakup_momentum",
    "covariant_spin_factor",
    "energy_dependent_width",
    "kallen",
]
