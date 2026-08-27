"""Composition of lineshape, barrier and angular factors."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.kinematics import covariant_kinematics

from .angular import CovariantAngular
from .context import ResonanceContext, resolve_value
from .lineshapes import (
    RelativisticBreitWigner,
    blatt_weisskopf_from_momenta,
    breakup_momentum,
)


def _key_index(key: str) -> int:
    try:
        return {"p1": 0, "p2": 1, "p3": 2}[key]
    except KeyError as exc:
        raise ValueError(
            "automatic identical-particle symmetrization requires p1/p2/p3 keys"
        ) from exc


def _identical_permutations(
    final_state: tuple[str, str, str],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        perm
        for perm in permutations(range(3))
        if all(final_state[i] == final_state[perm[i]] for i in range(3))
    )


@dataclass(frozen=True)
class ResonanceAmplitude:
    """Complete resonance amplitude assembled from interchangeable plugins."""

    context: ResonanceContext
    daughter_key: str = "p1"
    partner_key: str = "p2"
    bachelor_key: str = "p3"
    final_state: tuple[str, str, str] | None = None
    lineshape: object = RelativisticBreitWigner()
    angular: object = CovariantAngular()

    def _evaluate_pairing(
        self,
        data: Mapping[str, Array],
        daughter_key: str,
        partner_key: str,
        bachelor_key: str,
        parameters: Mapping[str, object] | None,
    ) -> Array:
        kin = covariant_kinematics(
            data[daughter_key], data[partner_key], data[bachelor_key]
        )
        context = self.context.resolve(parameters)
        lineshape = resolve_value(self.lineshape, parameters)
        angular_model = resolve_value(self.angular, parameters)
        l = int(context.spin)
        m1, m2 = context.daughter_masses

        q0 = breakup_momentum(context.pole_mass, m1, m2)
        p_star0 = breakup_momentum(
            context.parent_mass, context.pole_mass, context.bachelor_mass
        )
        x_res = blatt_weisskopf_from_momenta(
            kin.q, q0, l, context.resonance_radius
        )
        x_parent = blatt_weisskopf_from_momenta(
            kin.p_star, p_star0, l, context.parent_radius
        )
        resonance = lineshape(kin.resonance_mass, context)
        angular = angular_model(kin, context)
        return resonance * x_parent * x_res * angular

    def __call__(
        self,
        data: Mapping[str, Array],
        parameters: Mapping[str, object] | None = None,
    ) -> Array:
        base_keys = (self.daughter_key, self.partner_key, self.bachelor_key)
        if self.final_state is None:
            return self._evaluate_pairing(data, *base_keys, parameters)
        if len(self.final_state) != 3:
            raise ValueError("final_state must contain exactly three particle labels")

        role_indices = tuple(_key_index(key) for key in base_keys)
        seen = set()
        values = []
        for perm in _identical_permutations(self.final_state):
            keys = tuple(f"p{perm[index] + 1}" for index in role_indices)
            if keys in seen:
                continue
            seen.add(keys)
            values.append(self._evaluate_pairing(data, *keys, parameters))
        return sum(values, start=jnp.zeros_like(values[0]))
