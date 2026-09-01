"""Basic normalized PDFs for discriminating variables beyond the Dalitz plot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jax import Array

Parameters = Mapping[str, object]


def _resolve(value: object, parameters: Parameters | None):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


def jax_erf(x: Array) -> Array:
    from jax.scipy.special import erf as _erf
    return _erf(x)


@dataclass(frozen=True)
class Gaussian1D:
    """Gaussian PDF normalized on a finite interval."""

    mean: object
    sigma: object
    low: float
    high: float
    floor: float = 1e-300

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("Gaussian1D requires low < high")

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        mu = jnp.asarray(_resolve(self.mean, parameters))
        sigma = jnp.asarray(_resolve(self.sigma, parameters))
        z = (x - mu) / sigma
        raw = jnp.exp(-0.5 * z**2) / (jnp.sqrt(2.0 * jnp.pi) * sigma)
        a = (self.low - mu) / (jnp.sqrt(2.0) * sigma)
        b = (self.high - mu) / (jnp.sqrt(2.0) * sigma)
        norm = 0.5 * (jax_erf(b) - jax_erf(a))
        inside = (x >= self.low) & (x <= self.high) & (sigma > 0.0) & (norm > 0.0)
        return jnp.where(inside, jnp.clip(raw / norm, min=self.floor), 0.0)


@dataclass(frozen=True)
class Exponential1D:
    """Exponential PDF ``exp(slope*x)`` normalized on a finite interval."""

    slope: object
    low: float
    high: float
    floor: float = 1e-300

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("Exponential1D requires low < high")

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        slope = jnp.asarray(_resolve(self.slope, parameters))
        x = jnp.asarray(x)
        span = self.high - self.low
        near_zero = jnp.abs(slope) < 1e-10
        norm = jnp.where(
            near_zero,
            span,
            (jnp.exp(slope * self.high) - jnp.exp(slope * self.low)) / slope,
        )
        raw = jnp.exp(slope * x)
        inside = (x >= self.low) & (x <= self.high) & (norm > 0.0)
        return jnp.where(inside, jnp.clip(raw / norm, min=self.floor), 0.0)


@dataclass(frozen=True)
class Histogram1D:
    """Piecewise-constant normalized histogram PDF."""

    edges: Array
    values: Array

    def __post_init__(self) -> None:
        edges = jnp.asarray(self.edges)
        values = jnp.asarray(self.values)
        if edges.ndim != 1 or values.ndim != 1 or edges.size != values.size + 1:
            raise ValueError("Histogram1D requires len(edges)=len(values)+1")
        if bool(jnp.any(jnp.diff(edges) <= 0.0)) or bool(jnp.any(values < 0.0)):
            raise ValueError("Histogram1D requires increasing edges and non-negative values")
        widths = jnp.diff(edges)
        norm = jnp.sum(widths * values)
        if not bool(jnp.isfinite(norm)) or bool(norm <= 0.0):
            raise ValueError("Histogram1D integral must be positive and finite")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "values", values / norm)

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        idx = jnp.searchsorted(self.edges, x, side="right") - 1
        valid = (idx >= 0) & (idx < self.values.size)
        idx = jnp.clip(idx, 0, self.values.size - 1)
        return jnp.where(valid, self.values[idx], 0.0)


@dataclass(frozen=True)
class FactorizedDensity:
    """Multiply a base event density by independent discriminant PDFs.

    ``base_density(parameters)`` returns the Dalitz density evaluated on the
    fitted events. ``observables`` maps names to event arrays, and ``pdfs`` maps
    the same names to normalized one-dimensional PDF objects.
    """

    base_density: object
    observables: Mapping[str, Array]
    pdfs: Mapping[str, object]

    def __post_init__(self) -> None:
        if set(self.observables) != set(self.pdfs):
            raise ValueError("FactorizedDensity observables and pdfs must have identical keys")

    def __call__(self, parameters: Parameters) -> Array:
        base = jnp.asarray(self.base_density(parameters))
        result = base
        for name, values in self.observables.items():
            factor = jnp.asarray(self.pdfs[name](values, parameters))
            if factor.shape != base.shape:
                raise ValueError(f"discriminant {name!r} shape must match base density")
            result = result * factor
        return result


__all__ = ["Exponential1D", "FactorizedDensity", "Gaussian1D", "Histogram1D"]
