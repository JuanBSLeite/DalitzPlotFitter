"""Physics context shared by resonance dynamics plugins."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResonanceContext:
    """Kinematic and particle properties needed by resonance dynamics.

    The high-level :class:`DecayModel` constructs this object from the decay
    channel and the resonance declaration. Individual lineshapes and angular
    models consume only the fields they need.
    """

    parent_mass: float
    daughter_masses: tuple[float, float]
    bachelor_mass: float
    spin: int
    pole_mass: float
    pole_width: float
    resonance_radius: float = 1.5
    parent_radius: float = 5.0
