"""Numerical resonance dynamics."""

from .angular import CovariantAngular, covariant_spin_factor
from .context import ResonanceContext
from .lineshapes import (
    RelativisticBreitWigner,
    bachelor_momentum_resonance_frame,
    blatt_weisskopf_from_momenta,
    breakup_momentum,
    effective_pole_mass,
    energy_dependent_width,
    kallen,
)
from .resonance import ResonanceAmplitude

__all__ = [
    "CovariantAngular",
    "RelativisticBreitWigner",
    "ResonanceAmplitude",
    "ResonanceContext",
    "bachelor_momentum_resonance_frame",
    "blatt_weisskopf_from_momenta",
    "breakup_momentum",
    "covariant_spin_factor",
    "effective_pole_mass",
    "energy_dependent_width",
    "kallen",
]
