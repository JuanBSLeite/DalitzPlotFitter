"""Cached coherent amplitude evaluation for repeated likelihood calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.fit import Parameter, ParameterKind
from dalitzplotfitter.integration import matrix_normalization, normalization_matrix

from .components import AmplitudeComponent


def _component_scales(matrix: Array) -> Array:
    diagonal = jnp.real(jnp.diag(matrix))
    if bool(jnp.any(diagonal <= 0.0)):
        raise ValueError("Component normalization requires positive diagonal integrals")
    return 1.0 / jnp.sqrt(diagonal)


def _prepare_component_data(
    components: Sequence[AmplitudeComponent],
    data: Mapping[str, Array],
) -> Mapping[str, Array]:
    prepared: Mapping[str, Array] = dict(data)
    for component in components:
        prepare = getattr(component.function, "prepare_data", None)
        if prepare is not None:
            prepared = prepare(prepared)
    return prepared


@dataclass(frozen=True)
class PreparedAmplitudeCache:
    """Pre-evaluated amplitude components and normalization matrix."""

    components: tuple[AmplitudeComponent, ...]
    data: Mapping[str, Array]
    normalization_data: Mapping[str, Array]
    normalization_weights: Array
    parameters: tuple[Parameter, ...]
    data_components: Array
    normalization_components: Array
    normalization_matrix_fixed: Array
    efficiency_normalization: Array | None = None
    normalize_components: bool = True

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
        normalize_components: bool = True,
    ) -> "PreparedAmplitudeCache":
        components = tuple(components)
        if not components:
            raise ValueError("At least one amplitude component is required")

        # Parameter-independent event kinematics are prepared once here and then
        # reused by every dynamic lineshape evaluation during the fit.
        data = _prepare_component_data(components, data)
        normalization_data = _prepare_component_data(components, normalization_data)

        raw_data = jnp.stack(
            [jnp.asarray(c.function(data, None)) for c in components], axis=1
        )
        raw_norm = jnp.stack(
            [jnp.asarray(c.function(normalization_data, None)) for c in components],
            axis=1,
        )

        raw_component_matrix = normalization_matrix(
            raw_norm, normalization_weights, None
        )
        if normalize_components:
            scales = _component_scales(raw_component_matrix)
            data_components = raw_data * scales
            normalization_components = raw_norm * scales
        else:
            data_components = raw_data
            normalization_components = raw_norm

        fixed_matrix = normalization_matrix(
            normalization_components,
            normalization_weights,
            efficiency_normalization,
        )
        return cls(
            components=components,
            data=data,
            normalization_data=normalization_data,
            normalization_weights=jnp.asarray(normalization_weights),
            parameters=tuple(parameters),
            data_components=data_components,
            normalization_components=normalization_components,
            normalization_matrix_fixed=fixed_matrix,
            efficiency_normalization=efficiency_normalization,
            normalize_components=normalize_components,
        )

    @property
    def floating_dynamics(self) -> tuple[Parameter, ...]:
        return tuple(
            p
            for p in self.parameters
            if p.kind is ParameterKind.DYNAMICS and not p.fixed
        )

    @property
    def floating_dynamic_owners(self) -> frozenset[str]:
        return frozenset(
            p.owner for p in self.floating_dynamics if p.owner is not None
        )

    def coefficient_vector(self, fit_values: Mapping[str, object]) -> Array:
        return jnp.asarray(
            [component.coefficient.value(fit_values) for component in self.components]
        )

    def _dynamic_parameter_mapping(
        self,
        component_name: str,
        fit_values: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            p.backend_name or p.name: p.resolve(fit_values)
            for p in self.parameters
            if p.kind is ParameterKind.DYNAMICS and p.owner == component_name
        }

    def _pdf_weights(self) -> Array:
        weights = self.normalization_weights
        if self.efficiency_normalization is not None:
            weights = weights * jnp.asarray(self.efficiency_normalization)
        return weights

    def _single_component_scale(self, values: Array) -> Array:
        integral = jnp.mean(self.normalization_weights * jnp.abs(values) ** 2)
        return 1.0 / jnp.sqrt(integral)

    def _evaluate_components(
        self, fit_values: Mapping[str, object]
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
            data_values = jnp.asarray(component.function(self.data, pars))
            norm_values = jnp.asarray(component.function(self.normalization_data, pars))
            if self.normalize_components:
                scale = self._single_component_scale(norm_values)
                data_values = data_values * scale
                norm_values = norm_values * scale
            data_columns.append(data_values)
            norm_columns.append(norm_values)
        return jnp.stack(data_columns, axis=1), jnp.stack(norm_columns, axis=1)

    def _matrix_with_dynamic_blocks(self, norm_components: Array) -> Array:
        owners = self.floating_dynamic_owners
        if not owners:
            return self.normalization_matrix_fixed
        dynamic_indices = [
            i for i, component in enumerate(self.components) if component.name in owners
        ]
        weights = self._pdf_weights()
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

    def evaluate(self, fit_values: Mapping[str, object]) -> tuple[Array, Array]:
        coefficients = self.coefficient_vector(fit_values)
        data_components, norm_components = self._evaluate_components(fit_values)
        amplitude = data_components @ coefficients
        intensity = jnp.abs(amplitude) ** 2
        normalization = matrix_normalization(
            coefficients, self._matrix_with_dynamic_blocks(norm_components)
        )
        return intensity, normalization

    def amplitude(self, fit_values: Mapping[str, object]) -> Array:
        coefficients = self.coefficient_vector(fit_values)
        data_components, _ = self._evaluate_components(fit_values)
        return data_components @ coefficients

    def intensity(self, fit_values: Mapping[str, object]) -> Array:
        return jnp.abs(self.amplitude(fit_values)) ** 2

    def normalization(self, fit_values: Mapping[str, object]) -> Array:
        coefficients = self.coefficient_vector(fit_values)
        _, norm_components = self._evaluate_components(fit_values)
        return matrix_normalization(
            coefficients, self._matrix_with_dynamic_blocks(norm_components)
        )
