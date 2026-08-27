"""Weighted unbinned likelihood objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
from jax import Array

Parameters = Mapping[str, Array | float]
LogPDF = Callable[[dict[str, Array], Parameters], Array]


@dataclass(frozen=True)
class WeightedUnbinnedNLL:
    """Weighted objective ``-sum_i w_i log p(x_i)``.

    Negative finite weights are permitted, which is useful for diagnostic or
    externally weighted objectives. This class does *not* implement the
    covariance/error corrections required to quote statistically valid HESSE
    uncertainties for sPlot/sWeight fits.
    """

    logpdf: LogPDF
    data: dict[str, Array]
    weights: Array

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("weighted likelihood data must be non-empty")
        size = int(jnp.asarray(next(iter(self.data.values()))).shape[0])
        weights = jnp.asarray(self.weights)
        if weights.shape != (size,):
            raise ValueError(f"weights must have shape ({size},), got {weights.shape}")
        if not bool(jnp.all(jnp.isfinite(weights))):
            raise ValueError("weights must be finite")
        object.__setattr__(self, "weights", weights)

    def __call__(self, parameters: Parameters) -> Array:
        values = jnp.asarray(self.logpdf(self.data, parameters))
        if values.shape != self.weights.shape:
            raise ValueError(
                f"logpdf must return shape {self.weights.shape}, got {values.shape}"
            )
        return -jnp.sum(self.weights * values)
