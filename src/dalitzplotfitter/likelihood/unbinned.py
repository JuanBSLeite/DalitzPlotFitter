"""Unbinned negative log-likelihoods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
from jax import Array

Parameters = Mapping[str, Array | float]
LogPDF = Callable[[dict[str, Array], Parameters], Array]


@dataclass(frozen=True)
class UnbinnedNLL:
    """Standard unbinned negative log-likelihood."""

    logpdf: LogPDF
    data: dict[str, Array]

    def __call__(self, parameters: Parameters) -> Array:
        return -jnp.sum(self.logpdf(self.data, parameters))
