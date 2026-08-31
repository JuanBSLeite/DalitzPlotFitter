"""Kinematic utilities for three-body amplitudes."""

from .covariant import (
    CovariantKinematics,
    boost_to_rest_frame,
    covariant_kinematics,
    covariant_kinematics_from_invariants,
    spatial_magnitude,
)
from .dalitz_grid import DalitzGrid, dalitz_s13_limits
from .phase_space_mc import PhaseSpaceMC
from .sample import PhaseSpaceSample
from .square_dalitz import (
    SquareDalitzGrid,
    invariants_to_square_dalitz,
    square_dalitz_jacobian,
    square_dalitz_to_invariants,
)
from .vectors import invariant_mass_squared

__all__ = [
    "CovariantKinematics",
    "DalitzGrid",
    "PhaseSpaceMC",
    "PhaseSpaceSample",
    "SquareDalitzGrid",
    "boost_to_rest_frame",
    "covariant_kinematics",
    "covariant_kinematics_from_invariants",
    "dalitz_s13_limits",
    "invariant_mass_squared",
    "invariants_to_square_dalitz",
    "spatial_magnitude",
    "square_dalitz_jacobian",
    "square_dalitz_to_invariants",
]
