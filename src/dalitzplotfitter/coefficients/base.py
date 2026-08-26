"""Coefficient-set interfaces and CP-flavour convention."""

from __future__ import annotations

from enum import IntEnum
from typing import Protocol

from jax import Array


class Flavor(IntEnum):
    """Parent flavour used by CP-aware coefficient sets."""

    PARTICLE = +1
    ANTIPARTICLE = -1

    @property
    def sign(self) -> int:
        return int(self.value)


class Coefficient(Protocol):
    """Protocol implemented by all coefficient parameterisations."""

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        ...
