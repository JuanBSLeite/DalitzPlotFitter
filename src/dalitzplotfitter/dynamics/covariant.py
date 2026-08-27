"""Numerical Laura++ covariant resonance dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.kinematics import covariant_kinematics


def kallen(x, y, z):
    return x**2 + y**2 + z**2 - 2 * x * y - 2 * x * z - 2 * y * z


def breakup_momentum(mass, mass1, mass2):
    mass = jnp.asarray(mass)
    return jnp.sqrt(jnp.maximum(kallen(mass**2, mass1**2, mass2**2), 0.0)) / (2.0 * mass)


def _blatt_polynomial(z, angular_momentum: int):
    l = int(angular_momentum)
    if l == 0: return jnp.ones_like(z)
    if l == 1: return 1.0 + z**2
    if l == 2: return z**4 + 3.0 * z**2 + 9.0
    if l == 3: return z**6 + 6.0 * z**4 + 45.0 * z**2 + 225.0
    if l == 4: return z**8 + 10.0 * z**6 + 135.0 * z**4 + 1575.0 * z**2 + 11025.0
    raise NotImplementedError("Laura++ covariant components currently support L=0..4")


def blatt_weisskopf_from_momenta(momentum, pole_momentum, angular_momentum: int, radius):
    l = int(angular_momentum)
    momentum = jnp.asarray(momentum)
    if l == 0:
        return jnp.ones_like(momentum)
    z = momentum * radius
    z0 = pole_momentum * radius
    return jnp.sqrt(_blatt_polynomial(z0, l) / _blatt_polynomial(z, l))


def covariant_spin_factor(p_star, p, q, cos_theta, parent_mass, angular_momentum: int):
    l = int(angular_momentum)
    r = p**2 / parent_mass**2
    pq = p_star * q
    c = cos_theta
    if l == 0: return jnp.ones_like(c)
    if l == 1: return -2.0 * pq * jnp.sqrt(1.0 + r) * c
    if l == 2: return (4.0 / 3.0) * pq**2 * (1.5 + r) * (3.0 * c**2 - 1.0)
    if l == 3:
        return -(24.0 / 15.0) * pq**3 * jnp.sqrt(1.0 + r) * (2.5 + r) * (5.0 * c**3 - 3.0 * c)
    if l == 4:
        return (16.0 / 35.0) * pq**4 * (8.0 * r**2 + 40.0 * r + 35.0) * (35.0 * c**4 - 30.0 * c**2 + 3.0)
    raise NotImplementedError("Laura++ covariant angular factors support L=0..4")


def energy_dependent_width(mass, mass0, width0, daughter_mass1, daughter_mass2, angular_momentum: int, resonance_radius):
    l = int(angular_momentum)
    q = breakup_momentum(mass, daughter_mass1, daughter_mass2)
    q0 = breakup_momentum(mass0, daughter_mass1, daughter_mass2)
    x_res = blatt_weisskopf_from_momenta(q, q0, l, resonance_radius)
    safe_q0 = jnp.where(q0 > 0.0, q0, 1.0)
    safe_mass = jnp.where(mass > 0.0, mass, 1.0)
    return width0 * (q / safe_q0) ** (2 * l + 1) * (mass0 / safe_mass) * x_res**2


def relativistic_breit_wigner(mass, mass0, width0, daughter_mass1, daughter_mass2, angular_momentum: int, resonance_radius):
    width = energy_dependent_width(mass, mass0, width0, daughter_mass1, daughter_mass2, angular_momentum, resonance_radius)
    return 1.0 / (mass0**2 - mass**2 - 1j * mass0 * width)


def _key_index(key: str) -> int:
    try:
        return {"p1": 0, "p2": 1, "p3": 2}[key]
    except KeyError as exc:
        raise ValueError("automatic identical-particle symmetrization requires p1/p2/p3 keys") from exc


def _identical_permutations(final_state: tuple[str, str, str]) -> tuple[tuple[int, int, int], ...]:
    """Return permutations that exchange only identical final-state particles."""
    return tuple(
        perm for perm in permutations(range(3))
        if all(final_state[i] == final_state[perm[i]] for i in range(3))
    )


@dataclass(frozen=True)
class LauraCovariantRBW:
    """Complete Laura++ RBW component with automatic Bose symmetrization.

    ``F = R(m) X_L(p* r_parent) X_L(q r_res) T_L``.

    If ``final_state`` contains identical particles, all permutations that only
    exchange identical particles are summed coherently inside this resonance
    component. For ``("pi-", "pi+", "pi+")`` and the nominal pairing
    ``(p1,p2)p3`` this gives ``F[(12)3] + F[(13)2]`` automatically.
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
    final_state: tuple[str, str, str] | None = None

    @property
    def parameters(self) -> dict[str, object]:
        return {}

    def _evaluate_pairing(self, data: Mapping[str, Array], daughter_key: str, partner_key: str, bachelor_key: str) -> Array:
        kin = covariant_kinematics(data[daughter_key], data[partner_key], data[bachelor_key])
        l = int(self.angular_momentum)
        m1, m2 = self.daughter_masses
        q0 = breakup_momentum(self.mass0, m1, m2)
        p_star0 = breakup_momentum(self.parent_mass, self.mass0, self.bachelor_mass)
        x_res = blatt_weisskopf_from_momenta(kin.q, q0, l, self.resonance_radius)
        x_parent = blatt_weisskopf_from_momenta(kin.p_star, p_star0, l, self.parent_radius)
        resonance = relativistic_breit_wigner(kin.resonance_mass, self.mass0, self.width0, m1, m2, l, self.resonance_radius)
        angular = covariant_spin_factor(kin.p_star, kin.p, kin.q, kin.cos_theta, self.parent_mass, l)
        return resonance * x_parent * x_res * angular

    def __call__(self, data: Mapping[str, Array], parameters: Mapping[str, object] | None = None) -> Array:
        del parameters
        base_keys = (self.daughter_key, self.partner_key, self.bachelor_key)
        if self.final_state is None:
            return self._evaluate_pairing(data, *base_keys)
        if len(self.final_state) != 3:
            raise ValueError("final_state must contain exactly three particle labels")

        role_indices = tuple(_key_index(key) for key in base_keys)
        terms = []
        for perm in _identical_permutations(self.final_state):
            keys = tuple(f"p{perm[index] + 1}" for index in role_indices)
            if keys not in [item[0] for item in terms]:
                terms.append((keys, self._evaluate_pairing(data, *keys)))
        return sum((value for _, value in terms), start=jnp.zeros_like(terms[0][1]))
