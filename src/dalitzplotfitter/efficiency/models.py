"""Built-in efficiency models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax import Array


def _validate_histogram_edges(edges: Array, label: str) -> Array:
    edges = jnp.asarray(edges)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"{label} edges must be a one-dimensional array with at least two entries")
    if not bool(jnp.all(jnp.isfinite(edges))):
        raise ValueError(f"{label} edges must be finite")
    if not bool(jnp.all(jnp.diff(edges) > 0.0)):
        raise ValueError(f"{label} edges must be strictly increasing")
    return edges


@dataclass(frozen=True)
class UnityEfficiency:
    """Efficiency model corresponding to no efficiency correction."""

    def __call__(self, data: dict[str, Array]) -> Array:
        if not data:
            raise ValueError("efficiency evaluation requires non-empty event data")
        first = jnp.asarray(next(iter(data.values())))
        size = first.shape[0] if first.ndim > 0 else 1
        return jnp.ones((size,), dtype=float)


@dataclass(frozen=True)
class FunctionalEfficiency:
    """Wrap a JAX-compatible non-negative relative efficiency function."""

    function: Callable[[dict[str, Array]], Array]

    def __call__(self, data: dict[str, Array]) -> Array:
        values = jnp.asarray(self.function(data))
        valid = jnp.isfinite(values) & (values >= 0.0)
        return jnp.where(valid, values, jnp.nan)


@dataclass(frozen=True)
class HistogramEfficiency:
    """Piecewise-constant 2D relative efficiency histogram."""

    x_edges: Array
    y_edges: Array
    values: Array
    x_variable: str = "s12"
    y_variable: str = "s13"

    def __post_init__(self) -> None:
        x_edges = _validate_histogram_edges(self.x_edges, "x")
        y_edges = _validate_histogram_edges(self.y_edges, "y")
        values = jnp.asarray(self.values)
        expected = (x_edges.size - 1, y_edges.size - 1)
        if values.shape != expected:
            raise ValueError(f"Histogram values shape must be {expected}, got {values.shape}")
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("Histogram efficiency values must be finite")
        if bool(jnp.any(values < 0.0)):
            raise ValueError("Histogram efficiency values must be non-negative")
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
        values = self.values[safe_ix, safe_iy]
        return jnp.where(in_range, values, 0.0)
