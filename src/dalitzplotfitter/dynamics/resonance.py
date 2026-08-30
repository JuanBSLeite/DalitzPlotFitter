"""Composition of lineshape, barrier and angular factors."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.kinematics import (
    CovariantKinematics,
    covariant_kinematics,
    covariant_kinematics_from_invariants,
)

from .angular import CovariantAngular
from .context import ResonanceContext, resolve_value
from .lineshape import (
    RelativisticBreitWigner,
    bachelor_momentum_resonance_frame,
    blatt_weisskopf_from_momenta,
    breakup_momentum,
    effective_pole_mass,
)


def _key_index(key: str) -> int:
    try:
        return {"p1": 0, "p2": 1, "p3": 2}[key]
    except KeyError as exc:
        raise ValueError(
            "automatic identical-particle symmetrization requires p1/p2/p3 keys"
        ) from exc


def _invariant_key(first_key: str, second_key: str) -> str:
    pair = frozenset((_key_index(first_key), _key_index(second_key)))
    try:
        return {
            frozenset((0, 1)): "s12",
            frozenset((0, 2)): "s13",
            frozenset((1, 2)): "s23",
        }[pair]
    except KeyError as exc:
        raise ValueError("invalid particle-key pair") from exc


def _pair_mass(data: Mapping[str, Array], first_key: str, second_key: str) -> Array:
    key = _invariant_key(first_key, second_key)
    if key in data:
        return jnp.sqrt(jnp.maximum(jnp.asarray(data[key]), 0.0))
    first = jnp.asarray(data[first_key])
    second = jnp.asarray(data[second_key])
    pair = first + second
    mass2 = pair[..., 0] ** 2 - jnp.sum(pair[..., 1:] ** 2, axis=-1)
    return jnp.sqrt(jnp.maximum(mass2, 0.0))


def _kinematics(data, daughter_key, partner_key, bachelor_key, context):
    if all(key in data for key in (daughter_key, partner_key, bachelor_key)):
        return covariant_kinematics(
            data[daughter_key],
            data[partner_key],
            data[bachelor_key],
        )
    return covariant_kinematics_from_invariants(
        data[_invariant_key(daughter_key, partner_key)],
        data[_invariant_key(daughter_key, bachelor_key)],
        data[_invariant_key(partner_key, bachelor_key)],
        daughter_mass=context.daughter_masses[0],
        partner_mass=context.daughter_masses[1],
        bachelor_mass=context.bachelor_mass,
    )


@dataclass(frozen=True)
class ResonanceAmplitude:
    """One resonance amplitude including lineshape, barriers and angular term."""

    context: ResonanceContext
    daughter_key: str
    partner_key: str
    bachelor_key: str
    final_state: tuple[int, int, int]
    lineshape: object = RelativisticBreitWigner()
    angular: object = CovariantAngular()

    @property
    def parameters(self):
        from dalitzplotfitter.fit import Parameter

        found = {}
        for value in (
            self.context.pole_mass,
            self.context.pole_width,
            self.context.resonance_radius,
            self.context.parent_radius,
            self.lineshape,
            self.angular,
        ):
            if isinstance(value, Parameter):
                found[value.name] = value
            parameters = getattr(value, "parameters", None)
            if parameters is not None and not callable(parameters):
                try:
                    for parameter in parameters.values() if isinstance(parameters, dict) else parameters:
                        if isinstance(parameter, Parameter):
                            found[parameter.name] = parameter
                except TypeError:
                    pass
        return found

    def _single(self, data, context, daughter_key, partner_key, bachelor_key):
        kin = _kinematics(data, daughter_key, partner_key, bachelor_key, context)
        mass = _pair_mass(data, daughter_key, partner_key)
        m1, m2 = context.daughter_masses
        q = breakup_momentum(mass, m1, m2)
        q0 = breakup_momentum(effective_pole_mass(context), m1, m2)
        p = bachelor_momentum_resonance_frame(
            context.parent_mass, mass, context.bachelor_mass
        )
        p0 = bachelor_momentum_resonance_frame(
            context.parent_mass, effective_pole_mass(context), context.bachelor_mass
        )
        resonance_barrier = blatt_weisskopf_from_momenta(
            q, q0, context.spin, context.resonance_radius
        )
        parent_barrier = blatt_weisskopf_from_momenta(
            p, p0, context.spin, context.parent_radius
        )
        return (
            self.lineshape(mass, context)
            * resonance_barrier
            * parent_barrier
            * self.angular(kin, context.spin)
        )

    def __call__(self, data: Mapping[str, Array], parameters=None):
        context = resolve_value(self.context, parameters)
        lineshape = resolve_value(self.lineshape, parameters)
        angular = resolve_value(self.angular, parameters)
        base = ResonanceAmplitude(
            context=context,
            daughter_key=self.daughter_key,
            partner_key=self.partner_key,
            bachelor_key=self.bachelor_key,
            final_state=self.final_state,
            lineshape=lineshape,
            angular=angular,
        )
        total = base._single(
            data,
            context,
            self.daughter_key,
            self.partner_key,
            self.bachelor_key,
        )

        key_order = (self.daughter_key, self.partner_key, self.bachelor_key)
        id_order = tuple(self.final_state[_key_index(key)] for key in key_order)
        seen = {key_order}
        for perm in permutations(range(3)):
            permuted_ids = tuple(self.final_state[index] for index in perm)
            if permuted_ids != self.final_state:
                continue
            mapping = {"p1": f"p{perm[0] + 1}", "p2": f"p{perm[1] + 1}", "p3": f"p{perm[2] + 1}"}
            permuted = tuple(mapping[key] for key in key_order)
            if permuted in seen:
                continue
            permuted_id_order = tuple(self.final_state[_key_index(key)] for key in permuted)
            if permuted_id_order != id_order:
                continue
            seen.add(permuted)
            total = total + base._single(data, context, *permuted)
        return total
