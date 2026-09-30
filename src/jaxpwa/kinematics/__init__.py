"""Kinematic utilities for three-body and sequential multibody amplitudes."""

from .covariant import (
    CovariantKinematics,
    boost_to_rest_frame,
    covariant_kinematics,
    covariant_kinematics_from_invariants,
    spatial_magnitude,
)
from .dalitz_grid import dalitz_s13_limits
from .four_body import (
    cascade_coordinates,
    pair_coordinates,
    pair_coordinates_to_momenta,
)
from .nbody import NBodyPhaseSpaceMC, NBodySample
from .phase_space_mc import PhaseSpaceMC
from .sample import EventSample, PhaseSpaceSample
from .square_dalitz import (
    SquareDalitzGrid,
    fold_thetaprime,
    invariants_to_square_dalitz,
    square_dalitz_jacobian,
    square_dalitz_to_invariants,
)
from .vectors import invariant_mass_squared

__all__ = [
    "EventSample",
    "NBodyPhaseSpaceMC",
    "NBodySample",
    "cascade_coordinates",
    "pair_coordinates",
    "pair_coordinates_to_momenta",
    "CovariantKinematics",
    "PhaseSpaceMC",
    "PhaseSpaceSample",
    "SquareDalitzGrid",
    "boost_to_rest_frame",
    "covariant_kinematics",
    "covariant_kinematics_from_invariants",
    "dalitz_s13_limits",
    "fold_thetaprime",
    "invariant_mass_squared",
    "invariants_to_square_dalitz",
    "spatial_magnitude",
    "square_dalitz_jacobian",
    "square_dalitz_to_invariants",
]
