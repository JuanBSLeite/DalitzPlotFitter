"""Numerical Laura++ covariant resonance components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.kinematics import covariant_kinematics


def _kallen(x, y, z):
    return x**2 + y**2 + z**2 - 2 * x * y - 2 * x * z - 2 * y * z


def _breakup_momentum(mass, mass1, mass2):
    return jnp.sqrt(jnp.maximum(_kallen(mass**2, mass1**2, mass2**2), 0.0)) / (
        2.0 * mass
    )


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
    raise NotImplementedError("Covariant resonance components currently support L=0..4")


def blatt_weisskopf_from_momenta(
    momentum,
    pole_momentum,
    angular_momentum: int,
    radius,
):
    """Laura++ Blatt-Weisskopf factor with unit value at the pole momentum."""

    l = int(angular_momentum)
    momentum = jnp.asarray(momentum)
    if l == 0:
        return jnp.ones_like(momentum)
    z = momentum * radius
    z0 = pole_momentum * radius
    return jnp.sqrt(_blatt_polynomial(z0, l) / _blatt_polynomial(z, l))


def covariant_spin_factor(
    p_star,
    p,
    q,
    cos_theta,
    parent_mass,
    angular_momentum: int,
):
    """Numerical Laura++ covariant angular factor, Eqs. (91)-(95)."""

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
    raise NotImplementedError("Laura++ covariant angular factors support L=0..4")


@dataclass(frozen=True)
class LauraCovariantRBW:
    """Complete Laura++ RBW component with covariant angular dependence.

    The component evaluates

    ``F = R(m) X_L(p* r_parent) X_L(q r_res) T_L``

    using raw four-vectors already converted to the project's
    ``(E, px, py, pz)`` convention.
    """

    mass0: float
    width0: float
    parent_mass: float
    daughter_masses: tuple[float, float]
    bachelor_mass: float
    angular_momentum: int
    resonance_radius: float = 1.5
    parent_radius: float = 5.0
    daughter_key: str = "p1"
    partner_key: str = "p2"
    bachelor_key: str = "p3"

    @property
    def parameters(self) -> dict[str, object]:
        return {}

    def __call__(
        self,
        data: Mapping[str, Array],
        parameters: Mapping[str, object] | None = None,
    ) -> Array:
        del parameters
        kin = covariant_kinematics(
            data[self.daughter_key],
            data[self.partner_key],
            data[self.bachelor_key],
        )
        m = kin.resonance_mass
        l = int(self.angular_momentum)
        m1, m2 = self.daughter_masses

        q0 = _breakup_momentum(self.mass0, m1, m2)
        p_star0 = _breakup_momentum(
            self.parent_mass,
            self.mass0,
            self.bachelor_mass,
        )
        x_res = blatt_weisskopf_from_momenta(
            kin.q,
            q0,
            l,
            self.resonance_radius,
        )
        x_parent = blatt_weisskopf_from_momenta(
            kin.p_star,
            p_star0,
            l,
            self.parent_radius,
        )

        safe_q0 = jnp.where(q0 > 0.0, q0, 1.0)
        safe_m = jnp.where(m > 0.0, m, 1.0)
        running_width = (
            self.width0
            * (kin.q / safe_q0) ** (2 * l + 1)
            * (self.mass0 / safe_m)
            * x_res**2
        )
        resonance = 1.0 / (
            self.mass0**2 - m**2 - 1j * self.mass0 * running_width
        )
        angular = covariant_spin_factor(
            kin.p_star,
            kin.p,
            kin.q,
            kin.cos_theta,
            self.parent_mass,
            l,
        )
        return resonance * x_parent * x_res * angular
