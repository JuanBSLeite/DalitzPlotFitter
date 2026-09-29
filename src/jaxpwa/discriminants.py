"""Basic normalized PDFs for discriminating variables beyond the Dalitz plot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxpwa.dynamics.context import resolve_value

Parameters = Mapping[str, object]


def _resolve(value: object, parameters: Parameters | None):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


def jax_erf(x: Array) -> Array:
    from jax.scipy.special import erf as _erf
    return _erf(x)


def _gauss_legendre_nodes(low: float, high: float, order: int) -> tuple[Array, Array]:
    if high <= low:
        raise ValueError("integration interval requires low < high")
    if order < 2:
        raise ValueError("Gauss-Legendre order must be at least two")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    scale = 0.5 * (high - low)
    shift = 0.5 * (high + low)
    return jnp.asarray(scale * nodes + shift), jnp.asarray(scale * weights)


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
class BreitWigner1D:
    """Constant-width Breit-Wigner PDF normalized on a finite mass interval."""

    mean: object
    width: object
    low: float
    high: float
    floor: float = 1e-300

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("BreitWigner1D requires low < high")

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        mean = jnp.asarray(_resolve(self.mean, parameters))
        width = jnp.asarray(_resolve(self.width, parameters))
        gamma = 0.5 * width
        raw = gamma / ((x - mean) ** 2 + gamma**2)
        norm = jnp.arctan((self.high - mean) / gamma) - jnp.arctan(
            (self.low - mean) / gamma
        )
        inside = (
            (x >= self.low)
            & (x <= self.high)
            & (width > 0.0)
            & (norm > 0.0)
        )
        return jnp.where(inside, jnp.clip(raw / norm, min=self.floor), 0.0)


@dataclass(frozen=True)
class CrystalBall1D:
    """Crystal Ball PDF normalized on a finite interval.

    Follows the standard convention (matching ``scipy.stats.crystalball``):
    a Gaussian core for ``z = (x-mean)/sigma >= -alpha`` and a power-law tail
    on the left, ``alpha > 0`` and ``n > 1``. Normalization is closed-form —
    the Gaussian half via ``erf`` and the power-law half via its elementary
    antiderivative — rather than numerical quadrature, so it stays cheap
    under a JAX-compiled unbinned likelihood with many events.
    """

    mean: object
    sigma: object
    alpha: object
    n: object
    low: float
    high: float
    floor: float = 1e-300

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("CrystalBall1D requires low < high")

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        mu = jnp.asarray(_resolve(self.mean, parameters))
        sigma = jnp.asarray(_resolve(self.sigma, parameters))
        alpha = jnp.asarray(_resolve(self.alpha, parameters))
        n = jnp.asarray(_resolve(self.n, parameters))

        valid_shape = (sigma > 0.0) & (alpha > 0.0) & (n > 1.0)
        safe_sigma = jnp.where(valid_shape, sigma, 1.0)
        safe_alpha = jnp.where(valid_shape, alpha, 1.0)
        safe_n = jnp.where(valid_shape, n, 2.0)

        a_coeff = (safe_n / safe_alpha) ** safe_n * jnp.exp(-0.5 * safe_alpha**2)
        b_coeff = safe_n / safe_alpha - safe_alpha
        z_split = -safe_alpha

        def gauss_antideriv(z: Array) -> Array:
            return jnp.sqrt(0.5 * jnp.pi) * jax_erf(z / jnp.sqrt(2.0))

        def power_antideriv(z: Array) -> Array:
            base = jnp.clip(b_coeff - z, min=1e-300)
            return a_coeff * base ** (1.0 - safe_n) / (safe_n - 1.0)

        z_low = (self.low - mu) / safe_sigma
        z_high = (self.high - mu) / safe_sigma

        power_hi = jnp.minimum(z_high, z_split)
        power_norm = jnp.where(
            power_hi > z_low, power_antideriv(power_hi) - power_antideriv(z_low), 0.0
        )
        gauss_lo = jnp.maximum(z_low, z_split)
        gauss_norm = jnp.where(
            z_high > gauss_lo, gauss_antideriv(z_high) - gauss_antideriv(gauss_lo), 0.0
        )
        norm = power_norm + gauss_norm

        z = (x - mu) / safe_sigma
        base = jnp.clip(b_coeff - z, min=1e-300)
        power_pdf = a_coeff * base ** (-safe_n)
        gauss_pdf = jnp.exp(-0.5 * z**2)
        raw = jnp.where(z >= z_split, gauss_pdf, power_pdf) / safe_sigma

        inside = (x >= self.low) & (x <= self.high) & valid_shape & (norm > 0.0)
        return jnp.where(inside, jnp.clip(raw / norm, min=self.floor), 0.0)


@dataclass(frozen=True)
class LineshapeIntensity1D:
    """Normalize the intensity of an existing complex resonance lineshape.

    The wrapped lineshape follows ``lineshape(mass, ResonanceContext)`` and the
    returned PDF is ``|lineshape(m)|^2`` normalized on ``[low, high]``.
    ``from_context`` infers the full physical pair-mass interval,
    ``m1 + m2 <= m <= M_parent - m_bachelor``.

    ``quadrature_order`` is purely numerical and controls the Gauss--Legendre
    normalization grid. For a coherent model with interfering amplitudes,
    detector resolution must act on the full coherent intensity rather than on
    individual component intensities.
    """

    lineshape: object
    context: object
    low: float
    high: float
    quadrature_order: int = 256
    floor: float = 1e-300

    def __post_init__(self) -> None:
        nodes, weights = _gauss_legendre_nodes(
            self.low, self.high, self.quadrature_order
        )
        object.__setattr__(self, "_nodes", nodes)
        object.__setattr__(self, "_weights", weights)

    @classmethod
    def from_context(
        cls,
        lineshape: object,
        context: object,
        *,
        quadrature_order: int = 256,
        floor: float = 1e-300,
        parameters: Parameters | None = None,
    ) -> "LineshapeIntensity1D":
        """Use the full kinematic mass interval encoded by ``ResonanceContext``."""

        resolved = context.resolve(parameters)
        daughter1, daughter2 = resolved.daughter_masses
        low = float(daughter1 + daughter2)
        high = float(resolved.parent_mass - resolved.bachelor_mass)
        if high <= low:
            raise ValueError(
                "ResonanceContext has no physical two-body mass interval: "
                "parent_mass - bachelor_mass must exceed daughter mass sum"
            )
        return cls(
            lineshape=lineshape,
            context=context,
            low=low,
            high=high,
            quadrature_order=quadrature_order,
            floor=floor,
        )

    def _intensity(self, x: Array, parameters: Parameters | None = None) -> Array:
        context = self.context.resolve(parameters)
        lineshape = resolve_value(self.lineshape, parameters)
        amplitude = jnp.asarray(lineshape(jnp.asarray(x), context))
        return jnp.real(amplitude * jnp.conj(amplitude))

    def normalization(self, parameters: Parameters | None = None) -> Array:
        """Integral of ``|lineshape|^2`` over ``[low, high]`` by quadrature."""

        values = self._intensity(self._nodes, parameters)
        return jnp.sum(self._weights * values)

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        norm = self.normalization(parameters)
        intensity = self._intensity(x, parameters)
        inside = (x >= self.low) & (x <= self.high) & (norm > 0.0)
        return jnp.where(
            inside,
            jnp.clip(intensity / norm, min=self.floor),
            0.0,
        )


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
    """Multiply a base event density by independent discriminant PDFs."""

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


__all__ = [
    "BreitWigner1D",
    "CrystalBall1D",
    "Exponential1D",
    "FactorizedDensity",
    "Gaussian1D",
    "Histogram1D",
    "LineshapeIntensity1D",
]
