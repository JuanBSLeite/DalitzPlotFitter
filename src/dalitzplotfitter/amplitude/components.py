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
    normalize_component: bool | None = None

    def __post_init__(self) -> None:
        if self.normalize_component is not None and not isinstance(
            self.normalize_component, bool
        ):
            raise ValueError("normalize_component must be a boolean or None")

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
