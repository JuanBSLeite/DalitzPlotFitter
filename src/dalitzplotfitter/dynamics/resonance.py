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
    bachelor_momentum_parent_frame,
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
        raise ValueError("could not resolve invariant-mass key") from exc


def _kinematics_prefix(
    daughter_key: str,
    partner_key: str,
    bachelor_key: str,
) -> str:
    return f"__kin_{daughter_key}_{partner_key}_{bachelor_key}"


def _lineshape_prepared_key(prefix: str) -> str:
    return f"{prefix}_lineshape_prepared"


def _identical_permutations(
    final_state: tuple[str, str, str],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        perm
        for perm in permutations(range(3))
        if all(final_state[i] == final_state[perm[i]] for i in range(3))
    )


def _physical_pairings(
    final_state: tuple[str, str, str],
    role_indices: tuple[int, int, int],
    spin: int,
) -> tuple[tuple[str, str, str], ...]:
    daughter_index, partner_index, _ = role_indices
    identical_pair_daughters = (
        final_state[daughter_index] == final_state[partner_index]
    )
    if identical_pair_daughters and spin % 2:
        raise ValueError(
            "an odd-spin resonance cannot decay to two identical spinless bosons"
        )

    unique: dict[tuple[int, int, int], tuple[str, str, str]] = {}
    for perm in _identical_permutations(final_state):
        mapped = tuple(perm[index] for index in role_indices)
        if identical_pair_daughters:
            first, second, bachelor = mapped
            canonical = (*sorted((first, second)), bachelor)
        else:
            canonical = mapped
        unique.setdefault(
            canonical,
            tuple(f"p{index + 1}" for index in canonical),
        )
    return tuple(unique.values())


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
    bachelor_momentum_frame: str = "resonance"

    def __post_init__(self) -> None:
        if self.bachelor_momentum_frame not in {"resonance", "parent"}:
            raise ValueError(
                "bachelor_momentum_frame must be either 'resonance' or 'parent'"
            )

    def _pairings(self) -> tuple[tuple[str, str, str], ...]:
        base_keys = (self.daughter_key, self.partner_key, self.bachelor_key)
        if self.final_state is None:
            return (base_keys,)
        if len(self.final_state) != 3:
            raise ValueError("final_state must contain exactly three particle labels")
        role_indices = tuple(_key_index(key) for key in base_keys)
        return _physical_pairings(
            self.final_state,
            role_indices,
            int(self.context.spin),
        )

    def _kinematics(
        self,
        data: Mapping[str, Array],
        daughter_key: str,
        partner_key: str,
        bachelor_key: str,
        context: ResonanceContext,
    ) -> CovariantKinematics:
        prefix = _kinematics_prefix(daughter_key, partner_key, bachelor_key)
        prepared_keys = tuple(
            f"{prefix}_{name}"
            for name in ("mass", "pstar", "p", "q", "costheta")
        )
        if all(key in data for key in prepared_keys):
            return CovariantKinematics(
                resonance_mass=data[prepared_keys[0]],
                p_star=data[prepared_keys[1]],
                p=data[prepared_keys[2]],
                q=data[prepared_keys[3]],
                cos_theta=data[prepared_keys[4]],
            )

        resonance_key = _invariant_key(daughter_key, partner_key)
        daughter_bachelor_key = _invariant_key(daughter_key, bachelor_key)
        if resonance_key in data and daughter_bachelor_key in data:
            return covariant_kinematics_from_invariants(
                data[resonance_key],
                data[daughter_bachelor_key],
                parent_mass=context.parent_mass,
                daughter_mass=context.daughter_masses[0],
                partner_mass=context.daughter_masses[1],
                bachelor_mass=context.bachelor_mass,
            )
        return covariant_kinematics(
            data[daughter_key], data[partner_key], data[bachelor_key]
        )

    def prepare_data(self, data: Mapping[str, Array]) -> dict[str, Array]:
        """Attach parameter-independent kinematics and lineshape response data."""

        prepared = dict(data)
        context = self.context.resolve(None)
        prepare_lineshape = getattr(self.lineshape, "prepare_mass", None)
        for daughter_key, partner_key, bachelor_key in self._pairings():
            prefix = _kinematics_prefix(daughter_key, partner_key, bachelor_key)
            if f"{prefix}_mass" in prepared:
                kin = self._kinematics(
                    prepared,
                    daughter_key,
                    partner_key,
                    bachelor_key,
                    context,
                )
            else:
                kin = self._kinematics(
                    prepared,
                    daughter_key,
                    partner_key,
                    bachelor_key,
                    context,
                )
                prepared.update(
                    {
                        f"{prefix}_mass": kin.resonance_mass,
                        f"{prefix}_pstar": kin.p_star,
                        f"{prefix}_p": kin.p,
                        f"{prefix}_q": kin.q,
                        f"{prefix}_costheta": kin.cos_theta,
                    }
                )
            prepared_key = _lineshape_prepared_key(prefix)
            if prepare_lineshape is not None and prepared_key not in prepared:
                prepared[prepared_key] = prepare_lineshape(
                    kin.resonance_mass,
                    context,
                )
        return prepared

    def _evaluate_pairing(
        self,
        data: Mapping[str, Array],
        daughter_key: str,
        partner_key: str,
        bachelor_key: str,
        parameters: Mapping[str, object] | None,
    ) -> Array:
        context = self.context.resolve(parameters)
        kin = self._kinematics(
            data,
            daughter_key,
            partner_key,
            bachelor_key,
            context,
        )
        lineshape = resolve_value(self.lineshape, parameters)
        angular_model = resolve_value(self.angular, parameters)
        l = int(context.spin)
        m1, m2 = context.daughter_masses

        pole_mass_for_momenta = effective_pole_mass(context)
        q0 = breakup_momentum(pole_mass_for_momenta, m1, m2)
        if self.bachelor_momentum_frame == "parent":
            parent_momentum = kin.p_star
            p0 = bachelor_momentum_parent_frame(
                context.parent_mass,
                pole_mass_for_momenta,
                context.bachelor_mass,
            )
        else:
            parent_momentum = kin.p
            p0 = bachelor_momentum_resonance_frame(
                context.parent_mass,
                pole_mass_for_momenta,
                context.bachelor_mass,
            )
        x_res = blatt_weisskopf_from_momenta(
            kin.q, q0, l, context.resonance_radius
        )
        x_parent = blatt_weisskopf_from_momenta(
            parent_momentum, p0, l, context.parent_radius
        )

        prefix = _kinematics_prefix(daughter_key, partner_key, bachelor_key)
        prepared_key = _lineshape_prepared_key(prefix)
        evaluate_prepared = getattr(lineshape, "evaluate_prepared", None)
        if evaluate_prepared is not None and prepared_key in data:
            resonance = evaluate_prepared(
                kin.resonance_mass,
                data[prepared_key],
                context,
            )
        else:
            resonance = lineshape(kin.resonance_mass, context)

        angular = angular_model(kin, context)
        return resonance * x_parent * x_res * angular

    def __call__(
        self,
        data: Mapping[str, Array],
        parameters: Mapping[str, object] | None = None,
    ) -> Array:
        values = [
            self._evaluate_pairing(data, *keys, parameters)
            for keys in self._pairings()
        ]
        return sum(values, start=jnp.zeros_like(values[0]))
