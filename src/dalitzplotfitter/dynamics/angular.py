"""Angular models for resonance amplitudes."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .context import ResonanceContext


def _zemach_polynomial(cos_theta, angular_momentum: int):
    """Laura++ phase-convention polynomial entering Zemach spin factors."""
    l = int(angular_momentum)
    c = cos_theta
    if l == 0:
        return jnp.ones_like(c)
    if l == 1:
        return -2.0 * c
    if l == 2:
        return (4.0 / 3.0) * (3.0 * c**2 - 1.0)
    if l == 3:
        return -(24.0 / 15.0) * (5.0 * c**3 - 3.0 * c)
    if l == 4:
        return (16.0 / 35.0) * (35.0 * c**4 - 30.0 * c**2 + 3.0)
    raise NotImplementedError("Zemach angular factors support L=0..4")


def zemach_spin_factor(momentum, q, cos_theta, angular_momentum: int):
    """Return the Zemach spin factor using the selected bachelor momentum.

    The Laura++ ``Zemach_P`` and ``Zemach_Pstar`` conventions share the same
    polynomial and differ only in the bachelor momentum. ``Zemach_P`` uses
    the bachelor momentum in the resonance rest frame, while
    ``Zemach_Pstar`` uses it in the parent rest frame.
    """
    l = int(angular_momentum)
    return (momentum * q) ** l * _zemach_polynomial(cos_theta, l)


def covariant_spin_factor(p_star, p, q, cos_theta, parent_mass, angular_momentum: int):
    """Covariant angular factor following the conventions documented in Laura++."""
    l = int(angular_momentum)
    r = p**2 / parent_mass**2
    pq = p_star * q
    c = cos_theta
    if l == 0:
        return jnp.ones_like(c)
    if l == 1:
        return -2.0 * pq * jnp.sqrt(1.0 + r) * c
    if l == 2:
        return (4.0 / 3.0) * pq**2 * (1.5 + r) * (3.0 * c**2 - 1.0)
    if l == 3:
        return (
            -(24.0 / 15.0)
            * pq**3
            * jnp.sqrt(1.0 + r)
            * (2.5 + r)
            * (5.0 * c**3 - 3.0 * c)
        )
    if l == 4:
        return (
            (16.0 / 35.0)
            * pq**4
            * (8.0 * r**2 + 40.0 * r + 35.0)
            * (35.0 * c**4 - 30.0 * c**2 + 3.0)
        )
    raise NotImplementedError("covariant angular factors support L=0..4")


def goofit_legacy_spin_factor(p, q, cos_theta, angular_momentum: int):
    """Return the legacy GooFit spin factor used by the Ds->pipipi analysis.

    This reproduces the historical GooFit/Laura++ invariant convention after
    reduction to resonance-rest-frame momenta.  It is intentionally kept as a
    separate legacy option because its mass-dependent normalization is not the
    same as the current covariant or Zemach conventions.
    """
    l = int(angular_momentum)
    c = cos_theta
    pq = p * q
    if l == 0:
        return jnp.ones_like(c)
    if l == 1:
        return 4.0 * pq * c
    if l == 2:
        return (16.0 / 3.0) * pq**2 * (3.0 * c**2 - 1.0)
    raise NotImplementedError("GooFit legacy angular factors support L=0..2")


@dataclass(frozen=True)
class ZemachP:
    """Laura++ ``Zemach_P`` angular model.

    The bachelor momentum ``p`` is evaluated in the resonance rest frame.
    """

    def __call__(self, kinematics, context: ResonanceContext):
        return zemach_spin_factor(
            kinematics.p,
            kinematics.q,
            kinematics.cos_theta,
            context.spin,
        )


@dataclass(frozen=True)
class ZemachPstar:
    """Laura++ ``Zemach_Pstar`` angular model.

    The bachelor momentum ``p*`` is evaluated in the parent rest frame.
    """

    def __call__(self, kinematics, context: ResonanceContext):
        return zemach_spin_factor(
            kinematics.p_star,
            kinematics.q,
            kinematics.cos_theta,
            context.spin,
        )


# Laura++-style aliases are kept for users who want the formalism names exactly
# as they appear in amplitude-analysis documentation.
Zemach_P = ZemachP
Zemach_Pstar = ZemachPstar


@dataclass(frozen=True)
class CovariantAngular:
    """Default covariant angular model."""

    def __call__(self, kinematics, context: ResonanceContext):
        return covariant_spin_factor(
            kinematics.p_star,
            kinematics.p,
            kinematics.q,
            kinematics.cos_theta,
            context.parent_mass,
            context.spin,
        )


@dataclass(frozen=True)
class GooFitLegacyAngular:
    """Legacy angular convention used by the historical GooFit Ds->pipipi fit.

    The angular factor itself uses the bachelor momentum in the resonance rest
    frame.  To reproduce the complete historical GooFit resonance convention,
    construct ``ResonanceAmplitude`` with ``bachelor_momentum_frame="parent"``
    so that the parent Blatt--Weisskopf factor also follows the legacy setup.
    """

    def __call__(self, kinematics, context: ResonanceContext):
        return goofit_legacy_spin_factor(
            kinematics.p,
            kinematics.q,
            kinematics.cos_theta,
            context.spin,
        )


__all__ = [
    "CovariantAngular",
    "GooFitLegacyAngular",
    "ZemachP",
    "ZemachPstar",
    "Zemach_P",
    "Zemach_Pstar",
    "covariant_spin_factor",
    "goofit_legacy_spin_factor",
    "zemach_spin_factor",
]
