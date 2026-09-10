"""Normalized signal PDF."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.efficiency import UnityEfficiency
from dalitzplotfitter.integration import GridIntegrator

Parameters = Mapping[str, Array | float]
ParametricIntensity = Callable[[dict[str, Array], Parameters], Array]


def _normalized_log_density(numerator, normalization):
    """Keep physical zeros while avoiding invalid logarithms in autodiff."""
    valid_num = jnp.isfinite(numerator) & (numerator > 0)
    valid_norm = jnp.isfinite(normalization) & (normalization > 0)
    value = jnp.log(jnp.where(valid_num, numerator, 1.0)) - jnp.log(
        jnp.where(valid_norm, normalization, 1.0)
    )
    return jnp.where(valid_num & valid_norm, value, -jnp.inf)


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
    # Retained for constructor compatibility; physical zeros are never floored.
    floor: float = 1e-300

    def _acceptance(self, data: dict[str, Array]) -> Array:
        size = jnp.asarray(next(iter(data.values()))).shape[0]
        acceptance = jnp.ones((size,))
        for label, function in (("efficiency", self.efficiency), ("veto", self.veto)):
            if function is None:
                continue
            values = jnp.asarray(function(data))
            if values.ndim == 0:
                values = jnp.full((size,), values)
            if values.shape != (size,):
                raise ValueError(f"{label} must have shape ({size},)")
            valid = jnp.isfinite(values) & (values >= 0)
            acceptance = acceptance * jnp.where(valid, values, jnp.nan)
        return acceptance

    def normalization(self, parameters: Parameters) -> Array:
        return self.integrator.integrate(
            lambda data: self._acceptance(data) * self.intensity(data, parameters)
        )

    def __call__(self, data: dict[str, Array], parameters: Parameters) -> Array:
        numerator = self._acceptance(data) * self.intensity(data, parameters)
        normalization = self.normalization(parameters)
        valid = jnp.isfinite(numerator) & (numerator >= 0)
        valid_norm = jnp.isfinite(normalization) & (normalization > 0)
        density = numerator / jnp.where(valid_norm, normalization, 1.0)
        return jnp.where(valid & valid_norm, density, jnp.nan)

    def logpdf(self, data: dict[str, Array], parameters: Parameters) -> Array:
        numerator = self._acceptance(data) * self.intensity(data, parameters)
        return _normalized_log_density(numerator, self.normalization(parameters))
