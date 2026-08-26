"""Weighted unbinned likelihood, including sWeight use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
from jax import Array

Parameters = Mapping[str, Array | float]
LogPDF = Callable[[dict[str, Array], Parameters], Array]


@dataclass(frozen=True)
class WeightedUnbinnedNLL:
    """Weighted unbinned NLL: -sum_i w_i log p(x_i)."""

    logpdf: LogPDF
    data: dict[str, Array]
    weights: Array

    def __call__(self, parameters: Parameters) -> Array:
        return -jnp.sum(jnp.asarray(self.weights) * self.logpdf(self.data, parameters))
