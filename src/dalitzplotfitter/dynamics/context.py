"""Physics context shared by resonance dynamics plugins."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


def _resolve(value: object, values: Mapping[str, object] | None = None):
    resolver = getattr(value, "resolve", None)
    if resolver is None:
        return value
    if values is not None:
        backend_name = getattr(value, "backend_name", None)
        if backend_name is not None and backend_name in values:
            return values[backend_name]
    return resolver(values)


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
            parent_mass=_resolve(self.parent_mass, values),
            daughter_masses=tuple(
                _resolve(value, values) for value in self.daughter_masses
            ),
            bachelor_mass=_resolve(self.bachelor_mass, values),
            pole_mass=_resolve(self.pole_mass, values),
            pole_width=_resolve(self.pole_width, values),
            resonance_radius=_resolve(self.resonance_radius, values),
            parent_radius=_resolve(self.parent_radius, values),
        )
