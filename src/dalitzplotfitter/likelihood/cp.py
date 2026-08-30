"""Joint charge-Dalitz likelihood for direct-CP amplitude fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.amplitude import PreparedAmplitudeCache

Parameters = Mapping[str, Array | float]


@dataclass(frozen=True)
class CPJointNLL:
    """Unbinned CP likelihood with one normalization shared by both charges.

    For charge-conjugate amplitudes ``A_plus`` and ``A_minus`` the fitted
    probability density is defined over the joint space ``(Dalitz, charge)``:

    ``p(phi, q) = |A_q(phi)|^2 / (I_plus + I_minus)``.

    Consequently the observed relative numbers of positive and negative charge
    events contribute to the likelihood, rather than being conditioned away by
    normalising the two charge samples independently.
    """

    plus_cache: PreparedAmplitudeCache
    minus_cache: PreparedAmplitudeCache

    def __call__(self, parameters: Parameters) -> Array:
        intensity_plus, integral_plus = self.plus_cache.evaluate(parameters)
        intensity_minus, integral_minus = self.minus_cache.evaluate(parameters)

        tiny_plus = jnp.finfo(intensity_plus.dtype).tiny
        tiny_minus = jnp.finfo(intensity_minus.dtype).tiny
        total_integral = integral_plus + integral_minus
        total_events = intensity_plus.shape[0] + intensity_minus.shape[0]

        return (
            -jnp.sum(jnp.log(jnp.maximum(intensity_plus, tiny_plus)))
            -jnp.sum(jnp.log(jnp.maximum(intensity_minus, tiny_minus)))
            + total_events * jnp.log(total_integral)
        )

    def charge_probabilities(self, parameters: Parameters) -> tuple[Array, Array]:
        """Return the model probabilities for the positive and negative charges."""

        integral_plus = self.plus_cache.normalization(parameters)
        integral_minus = self.minus_cache.normalization(parameters)
        total = integral_plus + integral_minus
        return integral_plus / total, integral_minus / total


__all__ = ["CPJointNLL"]
