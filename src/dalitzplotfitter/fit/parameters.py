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
    """Configuration for one scalar fit parameter.

    ``backend_name`` is used when a user-facing parameter controls a parameter of
    an external numerical model, for example an AmpForm resonance mass or width.
    The public ``name`` remains stable even if the backend symbol is verbose.
    """

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

        if values is not None and self.name in values:
            return values[self.name]
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
        backend_name: str,
        owner: str,
        fixed: bool = False,
        bounds: tuple[float | None, float | None] | None = None,
        step: float | None = None,
    ) -> "Parameter":
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
        backend_name: str,
        owner: str,
        fixed: bool = True,
        bounds: tuple[float | None, float | None] | None = None,
        step: float | None = None,
    ) -> "Parameter":
        """Declare a Blatt-Weisskopf meson radius.

        Meson radii are fixed by default. Set ``fixed=False`` explicitly only when
        a fit is intended to vary the radius; in that case it is treated as a
        dynamical parameter and invalidates only the owning amplitude cache.
        """

        return cls.dynamics(
            name=name,
            value=value,
            backend_name=backend_name,
            owner=owner,
            fixed=fixed,
            bounds=bounds,
            step=step,
        )
