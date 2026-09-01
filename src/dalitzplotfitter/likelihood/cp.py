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

    The signal density is always normalized jointly over both charges,

    ``S_q(phi) = eps_q(phi) |A_q(phi)|^2 / (I_+ + I_-)``.

    If a background is supplied, its density is likewise normalized jointly,

    ``B_q(phi) = B_q^raw(phi) / (J_+ + J_-)``.

    Two mixture conventions are supported:

    * non-extended: ``p_q = f_sig S_q + (1-f_sig) B_q``;
    * extended: ``lambda_q = N_sig S_q + N_bkg B_q`` with
      ``NLL = N_sig + N_bkg - sum log(lambda_q)`` up to the usual
      parameter-independent factorial constant.

    In signal-only mode no mixture parameter is needed. Extended signal-only
    fits are supported with ``extended=True`` and ``signal_yield``.
    """

    plus_cache: PreparedAmplitudeCache
    minus_cache: PreparedAmplitudeCache
    plus_efficiency: Array | None = None
    minus_efficiency: Array | None = None
    plus_background: Array | None = None
    minus_background: Array | None = None
    plus_background_normalization: Array | float | None = None
    minus_background_normalization: Array | float | None = None
    signal_fraction: object | None = None
    extended: bool = False
    signal_yield: object | None = None
    background_yield: object | None = None

    def __post_init__(self) -> None:
        if (self.plus_efficiency is None) != (self.minus_efficiency is None):
            raise ValueError("plus_efficiency and minus_efficiency must be supplied together")

        background_shape_fields = (
            self.plus_background,
            self.minus_background,
            self.plus_background_normalization,
            self.minus_background_normalization,
        )
        supplied = tuple(value is not None for value in background_shape_fields)
        if any(supplied) and not all(supplied):
            raise ValueError(
                "background requires plus/minus background values and "
                "plus/minus background normalizations"
            )

        if self.extended:
            if self.signal_fraction is not None:
                raise ValueError("signal_fraction is not used in extended mode")
            if self.signal_yield is None:
                raise ValueError("extended mode requires signal_yield")
            if self.has_background and self.background_yield is None:
                raise ValueError("extended background fits require background_yield")
            if not self.has_background and self.background_yield is not None:
                raise ValueError("background_yield requires a background model")
        else:
            if self.signal_yield is not None or self.background_yield is not None:
                raise ValueError("signal_yield/background_yield require extended=True")
            if self.has_background and self.signal_fraction is None:
                raise ValueError("non-extended background fits require signal_fraction")
            if not self.has_background and self.signal_fraction is not None:
                raise ValueError("signal_fraction requires a background model")

    @property
    def has_background(self) -> bool:
        return self.plus_background is not None

    def _signal_densities(
        self, parameters: Parameters
    ) -> tuple[Array, Array, Array, Array]:
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

    def component_densities(
        self, parameters: Parameters
    ) -> tuple[tuple[Array, Array], tuple[Array, Array] | None]:
        """Return normalized signal and optional background charge densities."""

        signal_plus, signal_minus, _, _ = self._signal_densities(parameters)
        signal = (signal_plus, signal_minus)
        background = self._background_densities() if self.has_background else None
        return signal, background

    def densities(self, parameters: Parameters) -> tuple[Array, Array]:
        """Return the fitted non-extended PDF or extended event intensities."""

        (signal_plus, signal_minus), background = self.component_densities(parameters)

        if self.extended:
            n_signal = jnp.asarray(_resolve(self.signal_yield, parameters))
            if background is None:
                return n_signal * signal_plus, n_signal * signal_minus
            n_background = jnp.asarray(_resolve(self.background_yield, parameters))
            background_plus, background_minus = background
            return (
                n_signal * signal_plus + n_background * background_plus,
                n_signal * signal_minus + n_background * background_minus,
            )

        if background is None:
            return signal_plus, signal_minus

        f_signal = jnp.asarray(_resolve(self.signal_fraction, parameters))
        background_plus, background_minus = background
        return (
            f_signal * signal_plus + (1.0 - f_signal) * background_plus,
            f_signal * signal_minus + (1.0 - f_signal) * background_minus,
        )

    def expected_events(self, parameters: Parameters) -> Array:
        """Return the total expected yield for an extended fit."""

        if not self.extended:
            raise RuntimeError("expected_events is only defined in extended mode")
        total = jnp.asarray(_resolve(self.signal_yield, parameters))
        if self.has_background:
            total = total + jnp.asarray(_resolve(self.background_yield, parameters))
        return total

    def __call__(self, parameters: Parameters) -> Array:
        pdf_plus, pdf_minus = self.densities(parameters)
        tiny_plus = jnp.finfo(pdf_plus.dtype).tiny
        tiny_minus = jnp.finfo(pdf_minus.dtype).tiny
        nll = -jnp.sum(jnp.log(jnp.maximum(pdf_plus, tiny_plus))) - jnp.sum(
            jnp.log(jnp.maximum(pdf_minus, tiny_minus))
        )
        if self.extended:
            nll = nll + self.expected_events(parameters)
        return nll

    def charge_probabilities(self, parameters: Parameters) -> tuple[Array, Array]:
        """Return predicted positive/negative fractions after all components."""

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

        if self.extended:
            n_signal = jnp.asarray(_resolve(self.signal_yield, parameters))
            n_background = jnp.asarray(_resolve(self.background_yield, parameters))
            total_yield = n_signal + n_background
            return (
                (n_signal * signal_plus + n_background * background_plus) / total_yield,
                (n_signal * signal_minus + n_background * background_minus) / total_yield,
            )

        f_signal = jnp.asarray(_resolve(self.signal_fraction, parameters))
        return (
            f_signal * signal_plus + (1.0 - f_signal) * background_plus,
            f_signal * signal_minus + (1.0 - f_signal) * background_minus,
        )


__all__ = ["CPJointNLL"]
