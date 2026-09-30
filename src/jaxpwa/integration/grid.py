"""Deterministic Dalitz-grid integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from jaxpwa.kinematics.sample import EventSample


@dataclass(frozen=True)
class GridIntegrator:
    """Integrate a scalar event function on fixed weighted events of any dimension."""

    sample: EventSample

    def integrate(self, function: Callable[[dict[str, Array]], Array]) -> Array:
        values = jnp.asarray(function(self.sample.as_dict()))
        expected = self.sample.weights.shape
        if values.shape != expected:
            raise ValueError(
                f"Grid integrand must have shape {expected}, got {values.shape}"
            )
        return jnp.mean(self.sample.weights * values)
