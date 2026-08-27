"""Coherent amplitude components with RealImag coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class AmplitudeComponent:
    """Named Laura++ dynamical component ``F_i(x)`` with a RealImag coefficient."""

    name: str
    function: object
    coefficient: object

    def value(
        self,
        data: Mapping[str, Array],
        *,
        parameters: Mapping[str, object] | None = None,
        coefficient_values: Mapping[str, object] | None = None,
    ) -> Array:
        dynamics = jnp.asarray(self.function(data, parameters))
        coefficient = jnp.asarray(self.coefficient.value(coefficient_values))
        return coefficient * dynamics


@dataclass(frozen=True)
class ConstantAmplitude:
    """Non-resonant constant dynamical amplitude."""

    value_: complex = 1.0 + 0.0j

    @property
    def parameters(self) -> dict[str, object]:
        return {}

    def __call__(
        self,
        data: Mapping[str, Array],
        parameters: Mapping[str, object] | None = None,
    ) -> Array:
        del parameters
        if not data:
            return jnp.asarray(self.value_)
        first = jnp.asarray(next(iter(data.values())))
        size = first.shape[0] if first.ndim > 0 else 1
        return jnp.full((size,), self.value_, dtype=jnp.complex128)


@dataclass(frozen=True)
class BoseSymmetrizedAmplitude:
    """Coherently sum amplitudes related by exchange of identical particles.

    For ``D+ -> pi- pi+ pi+`` a resonant ``pi- pi+`` amplitude is evaluated in
    the two pairings ``(12)3`` and ``(13)2`` and summed before multiplication by
    its external complex coefficient, exactly as required by Bose symmetry.
    """

    first: object
    second: object

    @property
    def parameters(self) -> dict[str, object]:
        first_parameters = getattr(self.first, "parameters", {})
        second_parameters = getattr(self.second, "parameters", {})
        if first_parameters or second_parameters:
            raise NotImplementedError(
                "BoseSymmetrizedAmplitude currently expects fixed dynamical parameters"
            )
        return {}

    def __call__(
        self,
        data: Mapping[str, Array],
        parameters: Mapping[str, object] | None = None,
    ) -> Array:
        return jnp.asarray(self.first(data, parameters)) + jnp.asarray(
            self.second(data, parameters)
        )


@dataclass(frozen=True)
class CoherentAmplitudeModel:
    """Coherent sum ``A(x) = sum_i c_i F_i(x)``."""

    components: tuple[AmplitudeComponent, ...]

    def amplitude(
        self,
        data: Mapping[str, Array],
        *,
        parameters: Mapping[str, Mapping[str, object]] | None = None,
        coefficient_values: Mapping[str, object] | None = None,
    ) -> Array:
        if not self.components:
            raise ValueError("At least one amplitude component is required")
        total = None
        for component in self.components:
            component_parameters = None if parameters is None else parameters.get(component.name)
            value = component.value(
                data,
                parameters=component_parameters,
                coefficient_values=coefficient_values,
            )
            total = value if total is None else total + value
        return jnp.asarray(total)

    def intensity(self, data: Mapping[str, Array], **kwargs) -> Array:
        amplitude = self.amplitude(data, **kwargs)
        return jnp.real(amplitude * jnp.conj(amplitude))
