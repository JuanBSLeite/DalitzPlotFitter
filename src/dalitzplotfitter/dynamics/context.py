"""Physics context shared by resonance dynamics plugins."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Mapping


def resolve_value(value: object, values: Mapping[str, object] | None = None):
    """Resolve Parameters recursively inside simple plugin dataclasses."""

    resolver = getattr(value, "resolve", None)
    if resolver is not None:
        if values is not None:
            backend_name = getattr(value, "backend_name", None)
            if backend_name is not None and backend_name in values:
                return values[backend_name]
        return resolver(values)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: resolve_value(getattr(value, field.name), values)
            for field in fields(value)
        }
        return replace(value, **updates)
    if isinstance(value, tuple):
        return tuple(resolve_value(item, values) for item in value)
    if isinstance(value, list):
        return [resolve_value(item, values) for item in value]
    if isinstance(value, dict):
        return {key: resolve_value(item, values) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ResonanceContext:
    """Kinematic and particle properties needed by resonance dynamics.

    Scalar dynamical quantities may be numerical constants or fit ``Parameter``
    objects. ``resolve(values)`` returns a purely numerical context for the
    current likelihood evaluation.
    """

    parent_mass: object
    daughter_masses: tuple[object, object]
    bachelor_mass: object
    spin: int
    pole_mass: object
    pole_width: object
    resonance_radius: object = 1.5
    parent_radius: object = 5.0

    def resolve(
        self,
        values: Mapping[str, object] | None = None,
    ) -> "ResonanceContext":
        return replace(
            self,
            parent_mass=resolve_value(self.parent_mass, values),
            daughter_masses=tuple(
                resolve_value(value, values) for value in self.daughter_masses
            ),
            bachelor_mass=resolve_value(self.bachelor_mass, values),
            pole_mass=resolve_value(self.pole_mass, values),
            pole_width=resolve_value(self.pole_width, values),
            resonance_radius=resolve_value(self.resonance_radius, values),
            parent_radius=resolve_value(self.parent_radius, values),
        )
