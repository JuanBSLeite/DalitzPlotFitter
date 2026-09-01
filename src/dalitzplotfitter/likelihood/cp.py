"""Joint charge-Dalitz likelihoods for direct-CP amplitude fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.amplitude import PreparedAmplitudeCache
from dalitzplotfitter.background import CPBackgroundCategory

Parameters = Mapping[str, Array | float]


def _resolve(value: object, parameters: Parameters):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


@dataclass(frozen=True)
class CPJointNLL:
    """Unbinned likelihood over the joint ``(Dalitz, charge)`` sample space.

    The signal density is always normalized jointly over both charges,

    ``S_q(phi) = eps_q(phi) |A_q(phi)|^2 / (I_+ + I_-)``.

    Arbitrary named background categories can be supplied through
    ``background_categories``.  Each category is also normalized jointly over
    charge.  The legacy single-background arguments remain supported.

    Non-extended fits use a total ``signal_fraction`` and relative background
    composition fractions.  For ``N`` named background categories, the first
    ``N-1`` categories carry relative fractions and the final category is the
    remainder.  Extended fits use independent yields for signal and for every
    background category.
    """

    plus_cache: PreparedAmplitudeCache
    minus_cache: PreparedAmplitudeCache
    plus_efficiency: Array | None = None
    minus_efficiency: Array | None = None

    # Legacy single-background interface.
    plus_background: Array | None = None
    minus_background: Array | None = None
    plus_background_normalization: Array | float | None = None
    minus_background_normalization: Array | float | None = None

    # New arbitrary-category interface.
    background_categories: tuple[CPBackgroundCategory, ...] = ()

    signal_fraction: object | None = None
    extended: bool = False
    signal_yield: object | None = None
    background_yield: object | None = None

    def __post_init__(self) -> None:
        if (self.plus_efficiency is None) != (self.minus_efficiency is None):
            raise ValueError("plus_efficiency and minus_efficiency must be supplied together")

        legacy_fields = (
            self.plus_background,
            self.minus_background,
            self.plus_background_normalization,
            self.minus_background_normalization,
        )
        legacy_supplied = tuple(value is not None for value in legacy_fields)
        if any(legacy_supplied) and not all(legacy_supplied):
            raise ValueError(
                "background requires plus/minus background values and plus/minus normalizations"
            )
        if any(legacy_supplied) and self.background_categories:
            raise ValueError(
                "use either the legacy single-background arguments or background_categories, not both"
            )

        names = [category.name for category in self.background_categories]
        if len(set(names)) != len(names):
            raise ValueError("CP background category names must be unique")

        if self.extended:
            if self.signal_fraction is not None:
                raise ValueError("signal_fraction is not used in extended mode")
            if self.signal_yield is None:
                raise ValueError("extended mode requires signal_yield")
            if self.background_categories:
                if self.background_yield is not None:
                    raise ValueError(
                        "background_yield belongs to the legacy single-background interface"
                    )
                if any(category.fraction is not None for category in self.background_categories):
                    raise ValueError("background fractions are not used in extended mode")
                if any(category.yield_ is None for category in self.background_categories):
                    raise ValueError("every extended CP background category requires yield_")
            elif self.has_legacy_background:
                if self.background_yield is None:
                    raise ValueError("extended single-background fits require background_yield")
            elif self.background_yield is not None:
                raise ValueError("background_yield requires a background model")
        else:
            if self.signal_yield is not None or self.background_yield is not None:
                raise ValueError("signal_yield/background_yield require extended=True")
            if self.has_background and self.signal_fraction is None:
                raise ValueError("non-extended background fits require signal_fraction")
            if not self.has_background and self.signal_fraction is not None:
                raise ValueError("signal_fraction requires a background model")
            if self.background_categories:
                if any(category.yield_ is not None for category in self.background_categories):
                    raise ValueError("background yields require extended=True")
                if len(self.background_categories) > 1:
                    if any(c.fraction is None for c in self.background_categories[:-1]):
                        raise ValueError(
                            "all CP background categories except the last require a relative fraction"
                        )
                    if self.background_categories[-1].fraction is not None:
                        raise ValueError(
                            "the last CP background category is the remainder and must not define fraction"
                        )
                elif len(self.background_categories) == 1 and self.background_categories[0].fraction is not None:
                    raise ValueError(
                        "a single CP background category does not need a relative fraction"
                    )

    @property
    def has_legacy_background(self) -> bool:
        return self.plus_background is not None

    @property
    def has_background(self) -> bool:
        return self.has_legacy_background or bool(self.background_categories)

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

    def _legacy_background_densities(self) -> tuple[Array, Array]:
        if not self.has_legacy_background:
            raise RuntimeError("legacy background densities requested without legacy background")
        total = jnp.asarray(self.plus_background_normalization) + jnp.asarray(
            self.minus_background_normalization
        )
        return (
            jnp.asarray(self.plus_background) / total,
            jnp.asarray(self.minus_background) / total,
        )

    def background_weights(self, parameters: Parameters) -> Array:
        """Return relative weights for named background categories."""

        n = len(self.background_categories)
        if n == 0:
            return jnp.empty((0,), dtype=jnp.float64)
        if n == 1:
            return jnp.ones((1,), dtype=jnp.float64)
        explicit = jnp.asarray(
            [_resolve(category.fraction, parameters) for category in self.background_categories[:-1]],
            dtype=jnp.float64,
        )
        remainder = 1.0 - jnp.sum(explicit)
        return jnp.concatenate((explicit, jnp.asarray([remainder])))

    def densities(self, parameters: Parameters) -> tuple[Array, Array]:
        signal_plus, signal_minus, _, _ = self._signal_densities(parameters)

        if self.extended:
            n_signal = jnp.asarray(_resolve(self.signal_yield, parameters))
            total_plus = n_signal * signal_plus
            total_minus = n_signal * signal_minus
            if self.background_categories:
                for category in self.background_categories:
                    n_background = jnp.asarray(_resolve(category.yield_, parameters))
                    total_plus = total_plus + n_background * category.plus_density
                    total_minus = total_minus + n_background * category.minus_density
            elif self.has_legacy_background:
                n_background = jnp.asarray(_resolve(self.background_yield, parameters))
                background_plus, background_minus = self._legacy_background_densities()
                total_plus = total_plus + n_background * background_plus
                total_minus = total_minus + n_background * background_minus
            return total_plus, total_minus

        if not self.has_background:
            return signal_plus, signal_minus

        f_signal = jnp.asarray(_resolve(self.signal_fraction, parameters))
        if self.background_categories:
            weights = self.background_weights(parameters)
            background_plus = jnp.zeros_like(signal_plus)
            background_minus = jnp.zeros_like(signal_minus)
            for weight, category in zip(weights, self.background_categories):
                background_plus = background_plus + weight * category.plus_density
                background_minus = background_minus + weight * category.minus_density
        else:
            background_plus, background_minus = self._legacy_background_densities()

        return (
            f_signal * signal_plus + (1.0 - f_signal) * background_plus,
            f_signal * signal_minus + (1.0 - f_signal) * background_minus,
        )

    def expected_events(self, parameters: Parameters) -> Array:
        if not self.extended:
            raise RuntimeError("expected_events is only defined in extended mode")
        total = jnp.asarray(_resolve(self.signal_yield, parameters))
        if self.background_categories:
            for category in self.background_categories:
                total = total + jnp.asarray(_resolve(category.yield_, parameters))
        elif self.has_legacy_background:
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

        if self.extended:
            n_signal = jnp.asarray(_resolve(self.signal_yield, parameters))
            numerator_plus = n_signal * signal_plus
            numerator_minus = n_signal * signal_minus
            total_yield = n_signal
            if self.background_categories:
                for category in self.background_categories:
                    n_background = jnp.asarray(_resolve(category.yield_, parameters))
                    numerator_plus = numerator_plus + n_background * category.plus_probability
                    numerator_minus = numerator_minus + n_background * category.minus_probability
                    total_yield = total_yield + n_background
            else:
                n_background = jnp.asarray(_resolve(self.background_yield, parameters))
                bplus = jnp.asarray(self.plus_background_normalization)
                bminus = jnp.asarray(self.minus_background_normalization)
                btotal = bplus + bminus
                numerator_plus = numerator_plus + n_background * bplus / btotal
                numerator_minus = numerator_minus + n_background * bminus / btotal
                total_yield = total_yield + n_background
            return numerator_plus / total_yield, numerator_minus / total_yield

        f_signal = jnp.asarray(_resolve(self.signal_fraction, parameters))
        if self.background_categories:
            weights = self.background_weights(parameters)
            background_plus = jnp.asarray(0.0)
            background_minus = jnp.asarray(0.0)
            for weight, category in zip(weights, self.background_categories):
                background_plus = background_plus + weight * category.plus_probability
                background_minus = background_minus + weight * category.minus_probability
        else:
            bplus = jnp.asarray(self.plus_background_normalization)
            bminus = jnp.asarray(self.minus_background_normalization)
            btotal = bplus + bminus
            background_plus = bplus / btotal
            background_minus = bminus / btotal

        return (
            f_signal * signal_plus + (1.0 - f_signal) * background_plus,
            f_signal * signal_minus + (1.0 - f_signal) * background_minus,
        )


__all__ = ["CPJointNLL"]
