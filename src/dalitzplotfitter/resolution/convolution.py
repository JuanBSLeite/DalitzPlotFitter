"""Generic one-dimensional PDF convolution for detector-resolution effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import erf

Parameters = Mapping[str, object]


def _resolve(value: object, parameters: Parameters | None):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


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
class GaussianResolution1D:
    """Gaussian conditional resolution density ``R(x_obs | x_true)``.

    ``bias`` shifts the reconstructed mean relative to the true value and both
    ``bias`` and ``sigma`` may be ordinary numbers or fit ``Parameter`` objects.
    The kernel itself is normalized on the full real line; ``interval_probability``
    supplies the exact probability retained by a finite observed fit window.
    """

    sigma: object
    bias: object = 0.0
    floor: float = 0.0

    def __call__(
        self,
        observed: Array,
        true: Array,
        parameters: Parameters | None = None,
    ) -> Array:
        observed = jnp.asarray(observed)
        true = jnp.asarray(true)
        sigma = jnp.asarray(_resolve(self.sigma, parameters))
        bias = jnp.asarray(_resolve(self.bias, parameters))
        z = (observed - (true + bias)) / sigma
        raw = jnp.exp(-0.5 * z**2) / (jnp.sqrt(2.0 * jnp.pi) * sigma)
        valid = sigma > 0.0
        if self.floor > 0.0:
            raw = jnp.clip(raw, min=self.floor)
        return jnp.where(valid, raw, 0.0)

    def interval_probability(
        self,
        low: float,
        high: float,
        true: Array,
        parameters: Parameters | None = None,
    ) -> Array:
        """Probability for an event at ``true`` to reconstruct inside ``[low, high]``."""

        if high <= low:
            raise ValueError("observed interval requires low < high")
        true = jnp.asarray(true)
        sigma = jnp.asarray(_resolve(self.sigma, parameters))
        bias = jnp.asarray(_resolve(self.bias, parameters))
        root2_sigma = jnp.sqrt(2.0) * sigma
        mean = true + bias
        upper = (high - mean) / root2_sigma
        lower = (low - mean) / root2_sigma
        probability = 0.5 * (erf(upper) - erf(lower))
        return jnp.where(sigma > 0.0, probability, 0.0)


@dataclass(frozen=True)
class ConvolvedPDF1D:
    """Numerically convolve a normalized 1D PDF with a resolution kernel.

    The true PDF must follow the existing discriminant convention
    ``pdf(x, parameters)``. The kernel must provide both
    ``kernel(observed, true, parameters)`` and
    ``kernel.interval_probability(low, high, true, parameters)``.

    The numerator is integrated over ``[true_low, true_high]`` with
    Gauss--Legendre quadrature. The result is then normalized on the finite
    observed fit range ``[observed_low, observed_high]``. This normalization is
    parameter dependent whenever probability migrates across the observed
    boundaries.
    """

    pdf: object
    kernel: object
    true_low: float
    true_high: float
    observed_low: float
    observed_high: float
    order: int = 96
    floor: float = 1e-300

    def __post_init__(self) -> None:
        if self.true_high <= self.true_low:
            raise ValueError("ConvolvedPDF1D requires true_low < true_high")
        if self.observed_high <= self.observed_low:
            raise ValueError("ConvolvedPDF1D requires observed_low < observed_high")
        nodes, weights = _gauss_legendre_nodes(self.true_low, self.true_high, self.order)
        object.__setattr__(self, "_true_nodes", nodes)
        object.__setattr__(self, "_true_weights", weights)

    def normalization(self, parameters: Parameters | None = None) -> Array:
        nodes = self._true_nodes
        true_density = jnp.asarray(self.pdf(nodes, parameters))
        retained = jnp.asarray(
            self.kernel.interval_probability(
                self.observed_low,
                self.observed_high,
                nodes,
                parameters,
            )
        )
        return jnp.sum(self._true_weights * true_density * retained)

    def __call__(self, x: Array, parameters: Parameters | None = None) -> Array:
        x = jnp.asarray(x)
        nodes = self._true_nodes
        true_density = jnp.asarray(self.pdf(nodes, parameters))

        x_flat = jnp.ravel(x)
        response = jnp.asarray(
            self.kernel(x_flat[:, None], nodes[None, :], parameters)
        )
        numerator = jnp.sum(
            response * (self._true_weights * true_density)[None, :],
            axis=1,
        )
        norm = self.normalization(parameters)
        inside = (x_flat >= self.observed_low) & (x_flat <= self.observed_high) & (norm > 0.0)
        values = jnp.where(
            inside,
            jnp.clip(numerator / norm, min=self.floor),
            0.0,
        )
        return jnp.reshape(values, x.shape)


__all__ = ["ConvolvedPDF1D", "GaussianResolution1D"]
