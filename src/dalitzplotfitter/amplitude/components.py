"""Coherent amplitude components with DalitzPlotFitter-owned coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.coefficients import Flavor

from .model import CompiledModel, ParameterMapping


def _is_ampform_coupling(symbol: object) -> bool:
    """Identify AmpForm helicity couplings by their conventional C_ prefix."""

    return str(symbol).startswith("C_")


def compile_amplitude_component(model: object, *, use_cse: bool = True) -> CompiledModel:
    """Compile one complex AmpForm amplitude with its coupling fixed to unity.

    This function is intentionally different from :func:`compile_model`, which
    compiles the full intensity. Here AmpForm supplies only the complex dynamical
    function ``F_i(x)``. DalitzPlotFitter owns the external complex coefficient
    ``c_i`` that multiplies this component.

    The first implementation targets scalar three-body decays, for which one
    AmpForm model contains one helicity amplitude. More general spin configurations
    will require an explicit coherent/helicity grouping policy.
    """

    import sympy as sp
    from tensorwaves.function.sympy import create_parametrized_function

    if len(model.amplitudes) != 1:
        raise NotImplementedError(
            "External coefficient extraction currently requires exactly one "
            "AmpForm helicity amplitude"
        )

    expression = next(iter(model.amplitudes.values()))
    coupling_symbols = [
        symbol for symbol in model.parameter_defaults if _is_ampform_coupling(symbol)
    ]
    if not coupling_symbols:
        raise ValueError("No AmpForm coupling parameter was found in this component")

    expression = expression.xreplace({symbol: sp.Integer(1) for symbol in coupling_symbols})
    if hasattr(expression, "doit"):
        expression = expression.doit()

    parameters = {
        symbol: value
        for symbol, value in model.parameter_defaults.items()
        if symbol not in coupling_symbols
    }
    function = create_parametrized_function(
        expression=expression,
        parameters=parameters,
        backend="jax",
        use_cse=use_cse,
    )
    return CompiledModel(function)


@dataclass(frozen=True)
class AmplitudeComponent:
    """Named dynamical component ``F_i(x)`` with an external coefficient."""

    name: str
    function: object
    coefficient: object

    def value(
        self,
        data: Mapping[str, Array],
        *,
        flavor: Flavor = Flavor.PARTICLE,
        parameters: ParameterMapping | None = None,
    ) -> Array:
        dynamics = self.function(data, parameters)
        return jnp.asarray(self.coefficient.value(flavor)) * jnp.asarray(dynamics)


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
        parameters: ParameterMapping | None = None,
    ) -> Array:
        del parameters
        if not data:
            return jnp.asarray(self.value_)
        first = next(iter(data.values()))
        return jnp.full(jnp.asarray(first).shape, self.value_, dtype=jnp.complex128)


@dataclass(frozen=True)
class CoherentAmplitudeModel:
    """Coherent sum ``A(x) = sum_i c_i F_i(x)``."""

    components: tuple[AmplitudeComponent, ...]

    def amplitude(
        self,
        data: Mapping[str, Array],
        *,
        flavor: Flavor = Flavor.PARTICLE,
        parameters: Mapping[str, ParameterMapping] | None = None,
    ) -> Array:
        if not self.components:
            raise ValueError("At least one amplitude component is required")

        total = None
        for component in self.components:
            component_parameters = None
            if parameters is not None:
                component_parameters = parameters.get(component.name)
            value = component.value(
                data,
                flavor=flavor,
                parameters=component_parameters,
            )
            total = value if total is None else total + value
        return jnp.asarray(total)

    def intensity(
        self,
        data: Mapping[str, Array],
        *,
        flavor: Flavor = Flavor.PARTICLE,
        parameters: Mapping[str, ParameterMapping] | None = None,
    ) -> Array:
        amplitude = self.amplitude(data, flavor=flavor, parameters=parameters)
        return jnp.real(amplitude * jnp.conj(amplitude))

    def component_amplitudes(
        self,
        data: Mapping[str, Array],
        *,
        flavor: Flavor = Flavor.PARTICLE,
        parameters: Mapping[str, ParameterMapping] | None = None,
    ) -> dict[str, Array]:
        output = {}
        for component in self.components:
            component_parameters = None
            if parameters is not None:
                component_parameters = parameters.get(component.name)
            output[component.name] = component.value(
                data,
                flavor=flavor,
                parameters=component_parameters,
            )
        return output
