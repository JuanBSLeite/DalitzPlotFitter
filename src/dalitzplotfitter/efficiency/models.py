"""Built-in efficiency models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class UnityEfficiency:
    """Efficiency model corresponding to no efficiency correction."""

    def __call__(self, data: dict[str, Array]) -> Array:
        first = next(iter(data.values()))
        return jnp.ones_like(first, dtype=float)


@dataclass(frozen=True)
class FunctionalEfficiency:
    """Wrap an arbitrary JAX-compatible efficiency function."""

    function: Callable[[dict[str, Array]], Array]

    def __call__(self, data: dict[str, Array]) -> Array:
        values = jnp.asarray(self.function(data))
        return jnp.clip(values, min=0.0)


@dataclass(frozen=True)
class HistogramEfficiency:
    """Piecewise-constant 2D efficiency histogram."""

    x_edges: Array
    y_edges: Array
    values: Array
    x_variable: str = "s12"
    y_variable: str = "s13"

    def __post_init__(self) -> None:
        x_edges = jnp.asarray(self.x_edges)
        y_edges = jnp.asarray(self.y_edges)
        values = jnp.asarray(self.values)
        expected = (x_edges.size - 1, y_edges.size - 1)
        if values.shape != expected:
            raise ValueError(f"Histogram values shape must be {expected}, got {values.shape}")
        object.__setattr__(self, "x_edges", x_edges)
        object.__setattr__(self, "y_edges", y_edges)
        object.__setattr__(self, "values", values)

    def __call__(self, data: dict[str, Array]) -> Array:
        x = jnp.asarray(data[self.x_variable])
        y = jnp.asarray(data[self.y_variable])
        ix = jnp.searchsorted(self.x_edges, x, side="right") - 1
        iy = jnp.searchsorted(self.y_edges, y, side="right") - 1
        in_range = (
            (ix >= 0)
            & (ix < self.values.shape[0])
            & (iy >= 0)
            & (iy < self.values.shape[1])
        )
        safe_ix = jnp.clip(ix, 0, self.values.shape[0] - 1)
        safe_iy = jnp.clip(iy, 0, self.values.shape[1] - 1)
        values = jnp.clip(self.values[safe_ix, safe_iy], min=0.0)
        return jnp.where(in_range, values, 0.0)
