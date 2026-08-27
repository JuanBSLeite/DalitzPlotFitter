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

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("unbinned likelihood data must be non-empty")

    def __call__(self, parameters: Parameters) -> Array:
        size = int(jnp.asarray(next(iter(self.data.values()))).shape[0])
        values = jnp.asarray(self.logpdf(self.data, parameters))
        if values.shape != (size,):
            raise ValueError(f"logpdf must return shape ({size},), got {values.shape}")
        return -jnp.sum(values)
