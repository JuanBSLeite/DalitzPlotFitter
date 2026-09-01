"""Named background categories for multi-component likelihoods."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class BackgroundCategory:
    """One normalized-background category evaluated on the fitted data.

    ``values`` contains an unnormalized non-negative background shape at the
    data points and ``normalization`` is its integral over the accepted Dalitz
    region.  The category can optionally carry a non-extended fraction or an
    extended yield parameter.
    """

    name: str
    values: Array
    normalization: Array | float
    fraction: object | None = None
    yield_: object | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("background category name must be non-empty")
        values = jnp.asarray(self.values)
        if values.ndim != 1:
            raise ValueError("background category values must be one-dimensional")
        if bool(jnp.any(~jnp.isfinite(values))) or bool(jnp.any(values < 0.0)):
            raise ValueError("background category values must be finite and non-negative")
        normalization = jnp.asarray(self.normalization)
        if normalization.ndim != 0 or not bool(jnp.isfinite(normalization)) or bool(normalization <= 0.0):
            raise ValueError("background normalization must be a positive finite scalar")
        if self.fraction is not None and self.yield_ is not None:
            raise ValueError("a background category cannot define both fraction and yield")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "normalization", normalization)

    @property
    def density(self) -> Array:
        return self.values / self.normalization


__all__ = ["BackgroundCategory"]
