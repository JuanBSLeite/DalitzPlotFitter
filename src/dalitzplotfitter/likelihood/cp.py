"""Joint charge-Dalitz likelihoods for direct-CP amplitude fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.amplitude import PreparedAmplitudeCache

Parameters = Mapping[str, Array | float]


def _resolve(value: object, parameters: Parameters):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


@dataclass(frozen=True)
class CPJointNLL:
    """Unbinned likelihood over the joint ``(Dalitz, charge)`` sample space.

    In the signal-only case the two charge densities are

    ``p(phi,+) = |A_+(phi)|^2 / (I_+ + I_-)`` and
    ``p(phi,-) = |A_-(phi)|^2 / (I_+ + I_-)``.

    Optional event efficiencies may multiply the signal intensities. In that
    case the supplied caches must have been prepared with the corresponding
    efficiency values on their normalization samples, so that ``I_+`` and
    ``I_-`` are the efficiency-weighted signal integrals.

    A background mixture is enabled by supplying all of ``plus_background``,
    ``minus_background``, ``plus_background_normalization``,
    ``minus_background_normalization`` and ``background_fraction``. The
    background is normalized jointly over charge, exactly like the signal:

    ``p_q = (1-f) eps_q |A_q|^2/(I_+ + I_-) + f B_q/(J_+ + J_-)``.

    This preserves sensitivity to both local Dalitz CP asymmetries and the
    integrated relative rates of the two charge samples.
    """

    plus_cache: PreparedAmplitudeCache
    minus_cache: PreparedAmplitudeCache
    plus_efficiency: Array | None = None
    minus_efficiency: Array | None = None
    plus_background: Array | None = None
    minus_background: Array | None = None
    plus_background_normalization: Array | float | None = None
    minus_background_normalization: Array | float | None = None
    background_fraction: object | None = None

    def __post_init__(self) -> None:
        efficiencies = (self.plus_efficiency, self.minus_efficiency)
        if (efficiencies[0] is None) != (efficiencies[1] is None):
            raise ValueError("plus_efficiency and minus_efficiency must be supplied together")

        background_fields = (
            self.plus_background,
            self.minus_background,
            self.plus_background_normalization,
            self.minus_background_normalization,
            self.background_fraction,
        )
        supplied = tuple(value is not None for value in background_fields)
        if any(supplied) and not all(supplied):
            raise ValueError(
                "background mixture requires plus/minus background values, "
                "plus/minus background normalizations, and background_fraction"
            )

    @property
    def has_background(self) -> bool:
        return self.background_fraction is not None

    def _signal_densities(self, parameters: Parameters) -> tuple[Array, Array, Array, Array]:
        intensity_plus, integral_plus = self.plus_cache.evaluate(parameters)
        intensity_minus, integral_minus = self.minus_cache.evaluate(parameters)
        if self.plus_efficiency is not None:
            intensity_plus = jnp.asarray(self.plus_efficiency) * intensity_plus
            intensity_minus = jnp.asarray(self.minus_efficiency) * intensity_minus
        total_integral = integral_plus + integral_minus
        return (
            intensity_plus / total_integral,
            intensity_minus / total_integral,
            integral_plus,
            integral_minus,
        )

    def _background_densities(self) -> tuple[Array, Array]:
        if not self.has_background:
            raise RuntimeError("background densities requested for signal-only CPJointNLL")
        total = jnp.asarray(self.plus_background_normalization) + jnp.asarray(
            self.minus_background_normalization
        )
        return (
            jnp.asarray(self.plus_background) / total,
            jnp.asarray(self.minus_background) / total,
        )

    def densities(self, parameters: Parameters) -> tuple[Array, Array]:
        """Return normalized joint-space densities for the two charge samples."""

        signal_plus, signal_minus, _, _ = self._signal_densities(parameters)
        if not self.has_background:
            return signal_plus, signal_minus

        background_plus, background_minus = self._background_densities()
        fraction = jnp.asarray(_resolve(self.background_fraction, parameters))
        return (
            (1.0 - fraction) * signal_plus + fraction * background_plus,
            (1.0 - fraction) * signal_minus + fraction * background_minus,
        )

    def __call__(self, parameters: Parameters) -> Array:
        pdf_plus, pdf_minus = self.densities(parameters)
        tiny_plus = jnp.finfo(pdf_plus.dtype).tiny
        tiny_minus = jnp.finfo(pdf_minus.dtype).tiny
        return -jnp.sum(jnp.log(jnp.maximum(pdf_plus, tiny_plus))) - jnp.sum(
            jnp.log(jnp.maximum(pdf_minus, tiny_minus))
        )

    def charge_probabilities(self, parameters: Parameters) -> tuple[Array, Array]:
        """Return predicted total probabilities for positive and negative charge."""

        integral_plus = self.plus_cache.normalization(parameters)
        integral_minus = self.minus_cache.normalization(parameters)
        signal_total = integral_plus + integral_minus
        signal_plus = integral_plus / signal_total
        signal_minus = integral_minus / signal_total
        if not self.has_background:
            return signal_plus, signal_minus

        background_plus_norm = jnp.asarray(self.plus_background_normalization)
        background_minus_norm = jnp.asarray(self.minus_background_normalization)
        background_total = background_plus_norm + background_minus_norm
        background_plus = background_plus_norm / background_total
        background_minus = background_minus_norm / background_total
        fraction = jnp.asarray(_resolve(self.background_fraction, parameters))
        return (
            (1.0 - fraction) * signal_plus + fraction * background_plus,
            (1.0 - fraction) * signal_minus + fraction * background_minus,
        )


__all__ = ["CPJointNLL"]
