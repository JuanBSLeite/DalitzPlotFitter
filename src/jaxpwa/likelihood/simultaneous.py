"""Simultaneous likelihood composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from jax import Array

Parameters = Mapping[str, Array | float]
Objective = Callable[[Parameters], Array]


@dataclass(frozen=True)
class SimultaneousNLL:
    """Sum independent NLL terms sharing one parameter mapping."""

    terms: tuple[Objective, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("SimultaneousNLL requires at least one likelihood term")

    def __call__(self, parameters: Parameters) -> Array:
        return sum((term(parameters) for term in self.terms))
