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


def _component_scales(matrix: Array) -> Array:
    """Return factors that normalize each component intensity integral to one."""

    diagonal = jnp.real(jnp.diag(matrix))
    if bool(jnp.any(diagonal <= 0.0)):
        raise ValueError("Component normalization requires positive diagonal integrals")
    return 1.0 / jnp.sqrt(diagonal)


@dataclass(frozen=True)
class PreparedAmplitudeCache:
    """Pre-evaluated amplitudes for fast repeated likelihood evaluations.

    Coefficient-only fits reuse all line shapes on data and normalization samples,
    plus the complete normalization matrix. If dynamics float, only affected
    component columns and their normalization-matrix rows/columns are recomputed.

    With ``normalize_components=True``, each complete dynamical component is
    normalized on the reference normalization sample according to the Laura++
    convention

    ``integral epsilon(x) |F_i(x)|^2 dPhi = 1``.

    The normalization includes the full component supplied to the cache (line
    shape, angular term, barrier factors and symmetrization), not merely a 1D
    resonance mass function. For a floating dynamical component its individual
    normalization factor is recomputed whenever that component is reevaluated.
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
    normalize_components: bool = False
    component_scales_fixed: Array | None = None

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
        normalize_components: bool = False,
    ) -> "PreparedAmplitudeCache":
        components = tuple(components)
        if not components:
            raise ValueError("At least one amplitude component is required")

        raw_data_values = [jnp.asarray(c.function(data, None)) for c in components]
        raw_norm_values = [
            jnp.asarray(c.function(normalization_data, None)) for c in components
        ]
        raw_data_matrix = jnp.stack(raw_data_values, axis=1)
        raw_norm_matrix = jnp.stack(raw_norm_values, axis=1)
        raw_matrix = normalization_matrix(
            raw_norm_matrix,
            normalization_weights,
            efficiency_normalization,
        )

        scales = None
        data_matrix = raw_data_matrix
        norm_matrix_values = raw_norm_matrix
        fixed_matrix = raw_matrix
        if normalize_components:
            scales = _component_scales(raw_matrix)
            data_matrix = raw_data_matrix * scales
            norm_matrix_values = raw_norm_matrix * scales
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
            normalize_components=normalize_components,
            component_scales_fixed=scales,
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

    def _normalization_weights_with_efficiency(self) -> Array:
        weights = self.normalization_weights
        if self.efficiency_normalization is not None:
            weights = weights * jnp.asarray(self.efficiency_normalization)
        return weights

    def _single_component_scale(self, values: Array) -> Array:
        weights = self._normalization_weights_with_efficiency()
        integral = jnp.sum(weights * jnp.real(jnp.conj(values) * values)) / values.shape[0]
        return 1.0 / jnp.sqrt(integral)

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
            i for i, c in enumerate(self.components) if c.name in owners
        ]
        weights = self._normalization_weights_with_efficiency()
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
