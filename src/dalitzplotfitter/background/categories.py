"""Named background categories for multi-component likelihoods."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


def _validate_values(values, label: str) -> Array:
    values = jnp.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{label} values must be one-dimensional")
    if bool(jnp.any(~jnp.isfinite(values))) or bool(jnp.any(values < 0.0)):
        raise ValueError(f"{label} values must be finite and non-negative")
    return values


def _validate_normalization(value, label: str) -> Array:
    normalization = jnp.asarray(value)
    if normalization.ndim != 0 or not bool(jnp.isfinite(normalization)) or bool(normalization <= 0.0):
        raise ValueError(f"{label} normalization must be a positive finite scalar")
    return normalization


@dataclass(frozen=True)
class BackgroundCategory:
    """One normalized background category for a single data sample."""

    name: str
    values: Array
    normalization: Array | float
    fraction: object | None = None
    yield_: object | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("background category name must be non-empty")
        if self.fraction is not None and self.yield_ is not None:
            raise ValueError("a background category cannot define both fraction and yield")
        object.__setattr__(self, "values", _validate_values(self.values, self.name))
        object.__setattr__(
            self, "normalization", _validate_normalization(self.normalization, self.name)
        )

    @property
    def density(self) -> Array:
        return self.values / self.normalization


@dataclass(frozen=True)
class CPBackgroundCategory:
    """One background category in the joint ``(Dalitz, charge)`` space."""

    name: str
    plus_values: Array
    minus_values: Array
    plus_normalization: Array | float
    minus_normalization: Array | float
    fraction: object | None = None
    yield_: object | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("background category name must be non-empty")
        if self.fraction is not None and self.yield_ is not None:
            raise ValueError("a CP background category cannot define both fraction and yield")
        object.__setattr__(self, "plus_values", _validate_values(self.plus_values, f"{self.name} plus"))
        object.__setattr__(self, "minus_values", _validate_values(self.minus_values, f"{self.name} minus"))
        object.__setattr__(
            self,
            "plus_normalization",
            _validate_normalization(self.plus_normalization, f"{self.name} plus"),
        )
        object.__setattr__(
            self,
            "minus_normalization",
            _validate_normalization(self.minus_normalization, f"{self.name} minus"),
        )

    @property
    def normalization(self) -> Array:
        return self.plus_normalization + self.minus_normalization

    @property
    def plus_density(self) -> Array:
        return self.plus_values / self.normalization

    @property
    def minus_density(self) -> Array:
        return self.minus_values / self.normalization

    @property
    def plus_probability(self) -> Array:
        return self.plus_normalization / self.normalization

    @property
    def minus_probability(self) -> Array:
        return self.minus_normalization / self.normalization


__all__ = ["BackgroundCategory", "CPBackgroundCategory"]
