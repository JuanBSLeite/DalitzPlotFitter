"""Kinematic utilities for three-body amplitudes."""

from .covariant import (
    CovariantKinematics,
    boost_to_rest_frame,
    covariant_kinematics,
    covariant_kinematics_from_invariants,
    spatial_magnitude,
)
from .phase_space_mc import PhaseSpaceMC
from .sample import PhaseSpaceSample
from .vectors import invariant_mass_squared

__all__ = [
    "CovariantKinematics",
    "PhaseSpaceMC",
    "PhaseSpaceSample",
    "boost_to_rest_frame",
    "covariant_kinematics",
    "covariant_kinematics_from_invariants",
    "invariant_mass_squared",
    "spatial_magnitude",
]
