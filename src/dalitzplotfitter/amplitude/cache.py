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
    """Pre-evaluated amplitudes for fast repeated likelihood evaluations.

    Coefficient-only fits reuse all line shapes on data and normalization samples,
    plus the complete normalization matrix. If dynamics float, only affected
    component columns and their normalization-matrix rows/columns are recomputed.
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

        data_values = [jnp.asarray(c.function(data, None)) for c in components]
        norm_values = [jnp.asarray(c.function(normalization_data, None)) for c in components]
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
    def floating_dynamics(self) -> tuple[Parameter, ...]:
        return tuple(
            p for p in self.parameters
            if p.kind is ParameterKind.DYNAMICS and not p.fixed
        )

    @property
    def floating_dynamic_owners(self) -> frozenset[str]:
        return frozenset(
            p.owner for p in self.floating_dynamics if p.owner is not None
        )

    def coefficient_vector(
        self,
        fit_values: Mapping[str, object],
        *,
        flavor: Flavor = Flavor.PARTICLE,
    ) -> Array:
        return jnp.asarray([
            _coefficient_value(c.coefficient, flavor, fit_values)
            for c in self.components
        ])

    def _dynamic_parameter_mapping(
        self,
        component_name: str,
        fit_values: Mapping[str, object],
    ) -> dict[str, object]:
        output = {}
        for p in self.parameters:
            if p.kind is ParameterKind.DYNAMICS and p.owner == component_name:
                output[p.backend_name or p.name] = p.resolve(fit_values)
        return output

    def _evaluate_components(
        self,
        fit_values: Mapping[str, object],
    ) -> tuple[Array, Array]:
        owners = self.floating_dynamic_owners
        if not owners:
            return self.data_components, self.normalization_components

        data_columns = []
        norm_columns = []
        for i, component in enumerate(self.components):
            if component.name not in owners:
                data_columns.append(self.data_components[:, i])
                norm_columns.append(self.normalization_components[:, i])
                continue
            pars = self._dynamic_parameter_mapping(component.name, fit_values)
            data_columns.append(jnp.asarray(component.function(self.data, pars)))
            norm_columns.append(
                jnp.asarray(component.function(self.normalization_data, pars))
            )
        return jnp.stack(data_columns, axis=1), jnp.stack(norm_columns, axis=1)

    def _matrix_with_dynamic_blocks(self, norm_components: Array) -> Array:
        owners = self.floating_dynamic_owners
        if not owners:
            return self.normalization_matrix_fixed

        dynamic_indices = [
            i for i, c in enumerate(self.components) if c.name in owners
        ]
        weights = self.normalization_weights
        if self.efficiency_normalization is not None:
            weights = weights * jnp.asarray(self.efficiency_normalization)
        n_events = norm_components.shape[0]

        matrix = self.normalization_matrix_fixed
        for i in dynamic_indices:
            fi = norm_components[:, i]
            for j in range(norm_components.shape[1]):
                fj = norm_components[:, j]
                value = jnp.sum(weights * jnp.conj(fi) * fj) / n_events
                matrix = matrix.at[i, j].set(value)
                matrix = matrix.at[j, i].set(jnp.conj(value))
        return matrix

    def evaluate(
        self,
        fit_values: Mapping[str, object],
        *,
        flavor: Flavor = Flavor.PARTICLE,
    ) -> tuple[Array, Array]:
        """Return event intensity and normalization with one cache-aware pass."""

        coefficients = self.coefficient_vector(fit_values, flavor=flavor)
        data_components, norm_components = self._evaluate_components(fit_values)
        amplitude = data_components @ coefficients
        intensity = jnp.real(amplitude * jnp.conj(amplitude))
        matrix = self._matrix_with_dynamic_blocks(norm_components)
        normalization = matrix_normalization(coefficients, matrix)
        return intensity, normalization

    def amplitude(
        self,
        fit_values: Mapping[str, object],
        *,
        flavor: Flavor = Flavor.PARTICLE,
    ) -> Array:
        coefficients = self.coefficient_vector(fit_values, flavor=flavor)
        data_components, _ = self._evaluate_components(fit_values)
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
        _, norm_components = self._evaluate_components(fit_values)
        matrix = self._matrix_with_dynamic_blocks(norm_components)
        return matrix_normalization(coefficients, matrix)
