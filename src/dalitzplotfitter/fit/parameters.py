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

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must be non-empty")
        value = float(self.value)
        if self.step is not None and self.step <= 0.0:
            raise ValueError(f"parameter step must be positive for {self.name!r}")
        if self.bounds is None:
            return
        low, high = self.bounds
        if low is not None and high is not None and not low < high:
            raise ValueError(f"invalid bounds for {self.name!r}: {self.bounds}")
        if low is not None and value < low:
            raise ValueError(
                f"initial value {value} is below the lower bound {low} for {self.name!r}"
            )
        if high is not None and value > high:
            raise ValueError(
                f"initial value {value} is above the upper bound {high} for {self.name!r}"
            )

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

        if not owner:
            raise ValueError("dynamics parameters require a non-empty owner")
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
