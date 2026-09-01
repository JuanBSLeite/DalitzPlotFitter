"""Normalized signal PDF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.efficiency import UnityEfficiency
from dalitzplotfitter.integration import GridIntegrator

Parameters = Mapping[str, Array | float]
ParametricIntensity = Callable[[dict[str, Array], Parameters], Array]


@dataclass(frozen=True)
class SignalPDF:
    """Efficiency-corrected normalized signal density.

    An optional veto map is treated as a binary acceptance function and enters
    both the event numerator and the normalization integral.
    """

    intensity: ParametricIntensity
    integrator: GridIntegrator
    efficiency: object = UnityEfficiency()
    veto: object | None = None
    floor: float = 1e-300

    def _acceptance(self, data: dict[str, Array]) -> Array:
        efficiency = jnp.asarray(self.efficiency(data))
        if self.veto is None:
            return efficiency
        return efficiency * jnp.asarray(self.veto(data), dtype=efficiency.dtype)

    def normalization(self, parameters: Parameters) -> Array:
        return self.integrator.integrate(
            lambda data: self._acceptance(data) * self.intensity(data, parameters)
        )

    def __call__(self, data: dict[str, Array], parameters: Parameters) -> Array:
        numerator = self._acceptance(data) * self.intensity(data, parameters)
        return jnp.clip(numerator / self.normalization(parameters), min=self.floor)

    def logpdf(self, data: dict[str, Array], parameters: Parameters) -> Array:
        numerator = self._acceptance(data) * self.intensity(data, parameters)
        return jnp.log(jnp.clip(numerator, min=self.floor)) - jnp.log(
            self.normalization(parameters)
        )
