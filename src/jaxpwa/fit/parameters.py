"""Fit-parameter declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum


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
        """Declare a ``ParameterKind.COEFFICIENT`` parameter."""

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


def _fixed_parameter_updates(
    parameters: tuple[Parameter, ...],
    names: tuple[str, ...],
    values: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate a model-level fixing request and return target values."""
    by_name = {parameter.name: parameter for parameter in parameters}
    requested = set(names)
    if values is not None:
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping from parameter names to values")
        requested.update(values)
    if not requested:
        raise ValueError("supply at least one parameter name or value")
    if any(not isinstance(name, str) or not name for name in requested):
        raise ValueError("parameter names must be nonempty strings")
    unknown = sorted(requested.difference(by_name))
    if unknown:
        raise ValueError(f"unknown parameter name(s): {', '.join(unknown)}")

    updates = {name: float(by_name[name].value) for name in names}
    if values is not None:
        updates.update({name: float(value) for name, value in values.items()})
    return updates


def _replace_fixed_parameters(
    node: object,
    updates: Mapping[str, float],
    replacements: dict[str, Parameter] | None = None,
) -> object:
    """Recursively replace selected Parameters while preserving shared identity."""
    replacements = {} if replacements is None else replacements
    if isinstance(node, Parameter):
        if node.name not in updates:
            return node
        if node.name not in replacements:
            replacements[node.name] = replace(
                node, value=float(updates[node.name]), fixed=True
            )
        return replacements[node.name]
    replace_bindings = getattr(node, "_with_parameter_bindings", None)
    if callable(replace_bindings):
        bindings = node.parameters
        updated = _replace_fixed_parameters(bindings, updates, replacements)
        return node if updated is bindings else replace_bindings(updated)
    if is_dataclass(node) and not isinstance(node, type):
        changed = {}
        for field in fields(node):
            original = getattr(node, field.name)
            updated = _replace_fixed_parameters(original, updates, replacements)
            if updated is not original:
                if not field.init:
                    raise TypeError(
                        f"cannot replace Parameter in non-init field {field.name!r} "
                        f"of {type(node).__name__}"
                    )
                changed[field.name] = updated
        return node if not changed else replace(node, **changed)
    if isinstance(node, tuple):
        updated = tuple(
            _replace_fixed_parameters(item, updates, replacements) for item in node
        )
        return (
            node
            if all(a is b for a, b in zip(updated, node, strict=True))
            else updated
        )
    if isinstance(node, list):
        updated = [
            _replace_fixed_parameters(item, updates, replacements) for item in node
        ]
        return (
            node
            if all(a is b for a, b in zip(updated, node, strict=True))
            else updated
        )
    if isinstance(node, dict):
        updated = {
            key: _replace_fixed_parameters(item, updates, replacements)
            for key, item in node.items()
        }
        return (
            node
            if all(updated[key] is item for key, item in node.items())
            else updated
        )
    return node
