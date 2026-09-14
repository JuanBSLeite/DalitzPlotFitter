"""Built-in background models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.histogram import interpolate_2d


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
class FunctionalBackground:
    """Wrap a JAX-compatible non-negative background density."""

    function: Callable[[dict[str, Array]], Array]

    def __call__(self, data: dict[str, Array]) -> Array:
        values = jnp.asarray(self.function(data))
        valid = jnp.isfinite(values) & (values >= 0.0)
        return jnp.where(valid, values, jnp.nan)


@dataclass(frozen=True)
class HistogramBackground:
    """Piecewise-constant 2D background shape on Dalitz variables.

    ``folded=True`` folds the ``(x_variable, y_variable)`` pair onto
    ``x <= y`` before the bin lookup; see
    :class:`~dalitzplotfitter.efficiency.HistogramEfficiency` for the
    identical-daughter convention and the ``x_edges == y_edges`` requirement.
    """

    x_edges: Array
    y_edges: Array
    values: Array
    x_variable: str = "s12"
    y_variable: str = "s13"
    folded: bool = False
    interpolation: str = "none"

    def __post_init__(self) -> None:
        x_edges = _validate_histogram_edges(self.x_edges, "x")
        y_edges = _validate_histogram_edges(self.y_edges, "y")
        values = jnp.asarray(self.values)
        expected = (x_edges.size - 1, y_edges.size - 1)
        if values.shape != expected:
            raise ValueError(f"Histogram values shape must be {expected}, got {values.shape}")
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("Histogram background values must be finite")
        if bool(jnp.any(values < 0.0)):
            raise ValueError("Histogram background values must be non-negative")
        if self.folded and not bool(jnp.array_equal(x_edges, y_edges)):
            raise ValueError(
                "folded HistogramBackground requires x_edges and y_edges to be "
                "identical: folding looks min(x,y) up on the x grid and "
                "max(x,y) up on the y grid, so mismatched ranges would "
                "silently clamp whichever value is smaller/larger to the "
                "narrower grid's boundary"
            )
        if self.interpolation not in {"none", "linear", "spline"}:
            raise ValueError("interpolation must be 'none', 'linear' or 'spline'")
        object.__setattr__(self, "x_edges", x_edges)
        object.__setattr__(self, "y_edges", y_edges)
        object.__setattr__(self, "values", values)

    def __call__(self, data: dict[str, Array]) -> Array:
        x = jnp.asarray(data[self.x_variable])
        y = jnp.asarray(data[self.y_variable])
        if self.folded:
            x, y = jnp.minimum(x, y), jnp.maximum(x, y)
        # Cubic interpolation can overshoot below zero; the reference PDF
        # clamps such values before evaluating the likelihood.
        return jnp.maximum(interpolate_2d(x, y, self.x_edges, self.y_edges, self.values, self.interpolation), 0.0)

    def with_charge_asymmetry(self, normalization_sample, *, charge: int,
                              asymmetry: float = 0.0, veto=None):
        return charge_scaled_background(
            self, normalization_sample, charge=charge,
            asymmetry=asymmetry, veto=veto,
        )


@dataclass(frozen=True)
class ChargeScaledBackground:
    """Background shape with a prescribed B+/B- counting asymmetry.

    The wrapped shape is normalized on ``normalization_sample`` after
    ``veto`` and then multiplied by ``1 - charge * asymmetry``. This is the
    Laura++ convention for fixed background yields. ``generation_value`` is
    forwarded so Square-Dalitz backgrounds retain their separate raw-height
    generation path.
    """

    shape: object
    scale: float

    def __call__(self, data: dict[str, Array]) -> Array:
        return self.scale * jnp.asarray(self.shape(data))

    def generation_value(self, data: dict[str, Array]) -> Array:
        evaluator = getattr(self.shape, "generation_value", self.shape)
        return self.scale * jnp.asarray(evaluator(data))


def charge_scaled_background(shape, normalization_sample, *, veto=None,
                             asymmetry: float = 0.0, charge: int):
    """Return a shape normalized and scaled for one charge's yield."""
    if charge not in (-1, 1):
        raise ValueError("charge must be +1 or -1")
    data = normalization_sample.as_dict()
    acceptance = 1.0 if veto is None else jnp.asarray(veto(data))
    normalization = jnp.mean(normalization_sample.weights * acceptance * shape(data))
    if not bool(jnp.isfinite(normalization) & (normalization > 0.0)):
        raise ValueError("background must have a positive finite post-veto integral")
    return ChargeScaledBackground(shape, float((1.0 - charge * asymmetry) / normalization))
