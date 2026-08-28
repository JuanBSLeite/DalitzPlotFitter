"""Kinematic utilities for three-body amplitudes."""

from .covariant import (
    CovariantKinematics,
    boost_to_rest_frame,
    covariant_kinematics,
    covariant_kinematics_from_invariants,
    spatial_magnitude,
)
from .phasespace_mc import PhasespaceMC
from .sample import PhaseSpaceSample
from .vectors import invariant_mass_squared

__all__ = [
    "CovariantKinematics",
    "PhaseSpaceSample",
    "PhasespaceMC",
    "boost_to_rest_frame",
    "covariant_kinematics",
    "covariant_kinematics_from_invariants",
    "invariant_mass_squared",
    "spatial_magnitude",
]
