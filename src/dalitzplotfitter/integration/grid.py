"""Deterministic Dalitz-grid integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.kinematics import PhaseSpaceSample


@dataclass(frozen=True)
class GridIntegrator:
    """Integrate a scalar event function on a fixed Dalitz-grid sample."""

    sample: PhaseSpaceSample

    def integrate(self, function: Callable[[dict[str, Array]], Array]) -> Array:
        values = jnp.asarray(function(self.sample.as_dict()))
        expected = self.sample.weights.shape
        if values.shape != expected:
            raise ValueError(
                f"Grid integrand must have shape {expected}, got {values.shape}"
            )
        return jnp.mean(self.sample.weights * values)
