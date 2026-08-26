"""Cached coherent amplitude evaluation for repeated likelihood calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.coefficients import Flavor
from dalitzplotfitter.fit import Parameter, ParameterKind
from dalitzplotfitter.integration import matrix_normalization, normalization_matrix

from .components import AmplitudeComponent


def _coefficient_value(coefficient, flavor, values):
    try:
        return coefficient.value(flavor, values)
    except TypeError:
        return coefficient.value(flavor)


@dataclass(frozen=True)
class PreparedAmplitudeCache:
    """Pre-evaluated component amplitudes on data and normalization samples.

    Components without floating dynamics are reused verbatim for every likelihood
    evaluation. Components with floating dynamics are recomputed only when needed.
    Coefficient-only fits therefore never reevaluate line shapes or angular terms.
    """

    components: tuple[AmplitudeComponent, ...]
    data: Mapping[str, Array]
    normalization_data: Mapping[str, Array]
    normalization_weights: Array
    parameters: tuple[Parameter, ...]
    data_components: Array
    normalization_components: Array
    normalization_matrix_fixed: Array
    efficiency_normalization: Array | None = None

    @classmethod
    def prepare(
        cls,
        components: Sequence[AmplitudeComponent],
        *,
        data: Mapping[str, Array],
        normalization_data: Mapping[str, Array],
        normalization_weights: Array,
        parameters: Sequence[Parameter] = (),
        efficiency_normalization: Array | None = None,
    ) -> "PreparedAmplitudeCache":
        components = tuple(components)
        if not components:
            raise ValueError("At least one amplitude component is required")

        data_values = []
        norm_values = []
        for component in components:
            data_values.append(jnp.asarray(component.function(data, None)))
            norm_values.append(jnp.asarray(component.function(normalization_data, None)))

        data_matrix = jnp.stack(data_values, axis=1)
        norm_matrix_values = jnp.stack(norm_values, axis=1)
        fixed_matrix = normalization_matrix(
            norm_matrix_values,
            normalization_weights,
            efficiency_normalization,
        )
        return cls(
            components=components,
            data=data,
            normalization_data=normalization_data,
            normalization_weights=jnp.asarray(normalization_weights),
            parameters=tuple(parameters),
            data_components=data_matrix,
            normalization_components=norm_matrix_values,
            normalization_matrix_fixed=fixed_matrix,
            efficiency_normalization=efficiency_normalization,
        )

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(component.name for component in self.components)

    @property
    def floating_dynamics(self) -> tuple[Parameter, ...]:
        return tuple(
            parameter
            for parameter in self.parameters
            if parameter.kind is ParameterKind.DYNAMICS and not parameter.fixed
        )

    def coefficient_vector(
        self,
        fit_values: Mapping[str, object],
        *,
        flavor: Flavor = Flavor.PARTICLE,
    ) -> Array:
        return jnp.asarray([
            _coefficient_value(component.coefficient, flavor, fit_values)
            for component in self.components
        ])

    def _dynamic_parameter_mapping(
        self,
        component_name: str,
        fit_values: Mapping[str, object],
    ) -> dict[str, object]:
        output = {}
        for parameter in self.parameters:
            if parameter.kind is not ParameterKind.DYNAMICS:
                continue
            if parameter.owner != component_name:
                continue
            backend_name = parameter.backend_name or parameter.name
            output[backend_name] = parameter.resolve(fit_values)
        return output

    def component_matrices(
        self,
        fit_values: Mapping[str, object],
    ) -> tuple[Array, Array]:
        """Return data/MC component matrices, reusing all unaffected columns."""

        if not self.floating_dynamics:
            return self.data_components, self.normalization_components

        data_columns = []
        norm_columns = []
        floating_owners = {parameter.owner for parameter in self.floating_dynamics}
        for index, component in enumerate(self.components):
            if component.name not in floating_owners:
                data_columns.append(self.data_components[:, index])
                norm_columns.append(self.normalization_components[:, index])
                continue

            dynamic_parameters = self._dynamic_parameter_mapping(
                component.name,
                fit_values,
            )
            data_columns.append(
                jnp.asarray(component.function(self.data, dynamic_parameters))
            )
            norm_columns.append(
                jnp.asarray(
                    component.function(self.normalization_data, dynamic_parameters)
                )
            )

        return jnp.stack(data_columns, axis=1), jnp.stack(norm_columns, axis=1)

    def amplitude(
        self,
        fit_values: Mapping[str, object],
        *,
        flavor: Flavor = Flavor.PARTICLE,
    ) -> Array:
        coefficients = self.coefficient_vector(fit_values, flavor=flavor)
        data_components, _ = self.component_matrices(fit_values)
        return data_components @ coefficients

    def intensity(
        self,
        fit_values: Mapping[str, object],
        *,
        flavor: Flavor = Flavor.PARTICLE,
    ) -> Array:
        amplitude = self.amplitude(fit_values, flavor=flavor)
        return jnp.real(amplitude * jnp.conj(amplitude))

    def normalization(
        self,
        fit_values: Mapping[str, object],
        *,
        flavor: Flavor = Flavor.PARTICLE,
    ) -> Array:
        coefficients = self.coefficient_vector(fit_values, flavor=flavor)
        if not self.floating_dynamics:
            return matrix_normalization(coefficients, self.normalization_matrix_fixed)

        _, norm_components = self.component_matrices(fit_values)
        matrix = normalization_matrix(
            norm_components,
            self.normalization_weights,
            self.efficiency_normalization,
        )
        return matrix_normalization(coefficients, matrix)
