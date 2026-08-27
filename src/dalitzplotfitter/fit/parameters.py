"""Fit-parameter declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ParameterKind(str, Enum):
    """Role of a fit parameter inside the model."""

    COEFFICIENT = "coefficient"
    DYNAMICS = "dynamics"
    EFFICIENCY = "efficiency"
    BACKGROUND = "background"
    YIELD = "yield"
    OTHER = "other"


@dataclass(frozen=True)
class Parameter:
    """Configuration for one scalar fit parameter."""

    name: str
    value: float
    fixed: bool = False
    bounds: tuple[float | None, float | None] | None = None
    step: float | None = None
    kind: ParameterKind = ParameterKind.OTHER
    owner: str | None = None
    backend_name: str | None = None

    def resolve(self, values: Mapping[str, object] | None = None):
        """Return the current value from a flat fit-parameter mapping."""

        if values is not None:
            if self.name in values:
                return values[self.name]
            if self.backend_name is not None and self.backend_name in values:
                return values[self.backend_name]
        return self.value

    @classmethod
    def coefficient(
        cls,
        name: str,
        value: float,
        *,
        fixed: bool = False,
        bounds: tuple[float | None, float | None] | None = None,
        step: float | None = None,
        owner: str | None = None,
    ) -> "Parameter":
        return cls(
            name=name,
            value=value,
            fixed=fixed,
            bounds=bounds,
            step=step,
            kind=ParameterKind.COEFFICIENT,
            owner=owner,
        )

    @classmethod
    def dynamics(
        cls,
        name: str,
        value: float,
        *,
        owner: str,
        backend_name: str | None = None,
        fixed: bool = False,
        bounds: tuple[float | None, float | None] | None = None,
        step: float | None = None,
    ) -> "Parameter":
        """Declare a dynamical parameter owned by one amplitude component.

        ``backend_name`` is optional. When omitted the public parameter name is
        used directly throughout the numerical model.
        """

        return cls(
            name=name,
            value=value,
            fixed=fixed,
            bounds=bounds,
            step=step,
            kind=ParameterKind.DYNAMICS,
            owner=owner,
            backend_name=backend_name,
        )

    @classmethod
    def meson_radius(
        cls,
        name: str,
        value: float,
        *,
        owner: str,
        backend_name: str | None = None,
        fixed: bool = True,
        bounds: tuple[float | None, float | None] | None = None,
        step: float | None = None,
    ) -> "Parameter":
        """Declare a Blatt-Weisskopf meson radius."""

        return cls.dynamics(
            name=name,
            value=value,
            owner=owner,
            backend_name=backend_name,
            fixed=fixed,
            bounds=bounds,
            step=step,
        )
