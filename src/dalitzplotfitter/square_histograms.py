"""Histogram efficiency/background models defined in Square-Dalitz coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.kinematics import invariants_to_square_dalitz


def _validate_edges(edges: Array, label: str) -> Array:
    edges = jnp.asarray(edges)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"{label} edges must be one-dimensional with at least two entries")
    if not bool(jnp.all(jnp.isfinite(edges))) or not bool(jnp.all(jnp.diff(edges) > 0.0)):
        raise ValueError(f"{label} edges must be finite and strictly increasing")
    return edges


@dataclass(frozen=True)
class _SquareDalitzHistogram2D:
    mprime_edges: Array
    thetaprime_edges: Array
    values: Array
    mother_mass: float
    masses: tuple[float, float, float]
    pair: tuple[int, int] = (0, 1)

    def __post_init__(self) -> None:
        mp = _validate_edges(self.mprime_edges, "mprime")
        tp = _validate_edges(self.thetaprime_edges, "thetaprime")
        values = jnp.asarray(self.values)
        expected = (mp.size - 1, tp.size - 1)
        if values.shape != expected:
            raise ValueError(f"Square-Dalitz histogram values shape must be {expected}, got {values.shape}")
        if not bool(jnp.all(jnp.isfinite(values))) or bool(jnp.any(values < 0.0)):
            raise ValueError("Square-Dalitz histogram values must be finite and non-negative")
        if len(self.masses) != 3:
            raise ValueError("Square-Dalitz histograms require exactly three daughter masses")
        object.__setattr__(self, "mprime_edges", mp)
        object.__setattr__(self, "thetaprime_edges", tp)
        object.__setattr__(self, "values", values)

    def square_coordinates(self, data: dict[str, Array]) -> tuple[Array, Array]:
        missing = [key for key in ("s12", "s13", "s23") if key not in data]
        if missing:
            raise ValueError(f"Square-Dalitz histogram evaluation requires s12, s13 and s23; missing {missing}")
        return invariants_to_square_dalitz(
            data["s12"], data["s13"], data["s23"],
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )

    def __call__(self, data: dict[str, Array]) -> Array:
        mp, tp = self.square_coordinates(data)
        ix = jnp.searchsorted(self.mprime_edges, mp, side="right") - 1
        iy = jnp.searchsorted(self.thetaprime_edges, tp, side="right") - 1
        in_range = (
            (ix >= 0) & (ix < self.values.shape[0])
            & (iy >= 0) & (iy < self.values.shape[1])
        )
        safe_ix = jnp.clip(ix, 0, self.values.shape[0] - 1)
        safe_iy = jnp.clip(iy, 0, self.values.shape[1] - 1)
        return jnp.where(in_range, self.values[safe_ix, safe_iy], 0.0)


@dataclass(frozen=True)
class SquareDalitzHistogramEfficiency(_SquareDalitzHistogram2D):
    """Piecewise-constant efficiency map in ``(m', theta')``."""


@dataclass(frozen=True)
class SquareDalitzHistogramBackground(_SquareDalitzHistogram2D):
    """Piecewise-constant background shape in ``(m', theta')``."""


__all__ = ["SquareDalitzHistogramBackground", "SquareDalitzHistogramEfficiency"]
