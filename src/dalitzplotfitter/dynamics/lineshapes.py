"""Pluggable resonance lineshapes and barrier-factor helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .context import ResonanceContext


def kallen(x, y, z):
    return x**2 + y**2 + z**2 - 2 * x * y - 2 * x * z - 2 * y * z


def breakup_momentum(mass, mass1, mass2):
    """Two-body daughter momentum in the rest frame of ``mass``."""

    mass = jnp.asarray(mass)
    radicand = jnp.maximum(kallen(mass**2, mass1**2, mass2**2), 0.0)
    return jnp.sqrt(radicand) / (2.0 * mass)


def bachelor_momentum_resonance_frame(parent_mass, resonance_mass, bachelor_mass):
    """Bachelor momentum in the resonance rest frame.

    For ``P -> R b`` this is

    ``sqrt(lambda(m_P^2, m_R^2, m_b^2)) / (2 m_R)``.
    """

    resonance_mass = jnp.asarray(resonance_mass)
    radicand = jnp.maximum(
        kallen(parent_mass**2, resonance_mass**2, bachelor_mass**2), 0.0
    )
    return jnp.sqrt(radicand) / (2.0 * resonance_mass)


def effective_pole_mass(context: ResonanceContext):
    """Pole mass used only for momentum calculations of virtual states.

    A pole outside the physically accessible two-body range is smoothly mapped
    inside ``[m1+m2, m_parent-m_bachelor]``. The propagator itself continues to
    use the declared pole mass; only pole momenta use this effective value.
    """

    m0 = jnp.asarray(context.pole_mass)
    m1, m2 = context.daughter_masses
    minimum = jnp.asarray(m1 + m2)
    maximum = jnp.asarray(context.parent_mass - context.bachelor_mass)
    span = maximum - minimum
    midpoint = 0.5 * (minimum + maximum)
    mapped = minimum + 0.5 * span * (
        1.0 + jnp.tanh((m0 - midpoint) / span)
    )
    outside = (m0 < minimum) | (m0 > maximum)
    return jnp.where(outside, mapped, m0)


def _blatt_polynomial(z, angular_momentum: int):
    l = int(angular_momentum)
    if l == 0:
        return jnp.ones_like(z)
    if l == 1:
        return 1.0 + z**2
    if l == 2:
        return z**4 + 3.0 * z**2 + 9.0
    if l == 3:
        return z**6 + 6.0 * z**4 + 45.0 * z**2 + 225.0
    if l == 4:
        return z**8 + 10.0 * z**6 + 135.0 * z**4 + 1575.0 * z**2 + 11025.0
    raise NotImplementedError("barrier factors currently support L=0..4")


def blatt_weisskopf_from_momenta(
    momentum,
    pole_momentum,
    angular_momentum: int,
    radius,
):
    """Blatt-Weisskopf factor normalized to unity at the pole momentum."""

    l = int(angular_momentum)
    momentum = jnp.asarray(momentum)
    if l == 0:
        return jnp.ones_like(momentum)
    z = momentum * radius
    z0 = pole_momentum * radius
    return jnp.sqrt(_blatt_polynomial(z0, l) / _blatt_polynomial(z, l))


def energy_dependent_width(mass, context: ResonanceContext):
    """Mass-dependent width used by the default relativistic Breit-Wigner."""

    l = int(context.spin)
    m1, m2 = context.daughter_masses
    q = breakup_momentum(mass, m1, m2)
    q0 = breakup_momentum(effective_pole_mass(context), m1, m2)
    x_res = blatt_weisskopf_from_momenta(
        q, q0, l, context.resonance_radius
    )
    safe_mass = jnp.where(mass > 0.0, mass, jnp.nan)
    safe_q0 = jnp.where(q0 > 0.0, q0, jnp.nan)
    return (
        context.pole_width
        * (q / safe_q0) ** (2 * l + 1)
        * (context.pole_mass / safe_mass)
        * x_res**2
    )


@dataclass(frozen=True)
class RelativisticBreitWigner:
    """Relativistic Breit-Wigner lineshape with running width."""

    def __call__(self, mass, context: ResonanceContext):
        width = energy_dependent_width(mass, context)
        m0 = context.pole_mass
        return 1.0 / (m0**2 - mass**2 - 1j * m0 * width)
