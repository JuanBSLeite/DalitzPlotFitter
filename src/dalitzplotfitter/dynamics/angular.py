"""Angular models for resonance amplitudes."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .context import ResonanceContext


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
    """Legacy GooFit/Laura++ spin factor used in the Ds->pipipi analysis.

    In the resonance rest frame the explicit invariant formula used by the old
    GooFit branch reduces to these expressions. The convention is intentionally
    kept separate from the current default covariant angular model because its
    normalization is mass dependent and therefore cannot be absorbed into one
    constant complex coefficient.
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
    raise NotImplementedError("GooFit legacy angular factors currently support L=0..2")


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
    """Angular convention used by the legacy DsPPP GooFit implementation.

    ``parent_barrier_frame`` is consumed by ``ResonanceAmplitude`` so the parent
    Blatt--Weisskopf factor uses the bachelor momentum in the parent rest frame,
    matching the old analysis implementation.
    """

    parent_barrier_frame: str = "parent"

    def __call__(self, kinematics, context: ResonanceContext):
        return goofit_legacy_spin_factor(
            kinematics.p,
            kinematics.q,
            kinematics.cos_theta,
            context.spin,
        )
