"""Normalized signal PDF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.efficiency import UnityEfficiency
from dalitzplotfitter.integration import MonteCarloIntegrator

Parameters = Mapping[str, Array | float]
ParametricIntensity = Callable[[dict[str, Array], Parameters], Array]


@dataclass(frozen=True)
class SignalPDF:
    """Efficiency-corrected normalized signal density."""

    intensity: ParametricIntensity
    integrator: MonteCarloIntegrator
    efficiency: object = UnityEfficiency()
    floor: float = 1e-300

    def normalization(self, parameters: Parameters) -> Array:
        return self.integrator.integrate(
            lambda data: self.efficiency(data) * self.intensity(data, parameters)
        )

    def __call__(self, data: dict[str, Array], parameters: Parameters) -> Array:
        numerator = self.efficiency(data) * self.intensity(data, parameters)
        return jnp.clip(numerator / self.normalization(parameters), min=self.floor)

    def logpdf(self, data: dict[str, Array], parameters: Parameters) -> Array:
        numerator = self.efficiency(data) * self.intensity(data, parameters)
        return jnp.log(jnp.clip(numerator, min=self.floor)) - jnp.log(
            self.normalization(parameters)
        )
