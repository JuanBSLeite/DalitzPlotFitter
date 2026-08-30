"""Numerical resonance dynamics."""

from .angular import CovariantAngular, covariant_spin_factor
from .context import ResonanceContext
from .kmatrix import KMatrix
from .lineshapes import (
    Flatte,
    GounarisSakurai,
    RelativisticBreitWigner,
    bachelor_momentum_resonance_frame,
    blatt_weisskopf_from_momenta,
    breakup_momentum,
    effective_pole_mass,
    energy_dependent_width,
    kallen,
)
from .pole_lass import LASS, Pole
from .qmi import QMI
from .resonance import ResonanceAmplitude

__all__ = [
    "CovariantAngular",
    "Flatte",
    "GounarisSakurai",
    "KMatrix",
    "LASS",
    "Pole",
    "QMI",
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
