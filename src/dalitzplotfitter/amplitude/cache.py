"""Cached coherent amplitude evaluation for repeated likelihood calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.fit import Parameter, ParameterKind
from dalitzplotfitter.integration import matrix_normalization, normalization_matrix
from dalitzplotfitter.observables import fit_fractions as matrix_fit_fractions
from dalitzplotfitter.observables import (
    interference_fractions as matrix_interference_fractions,
)

from .components import AmplitudeComponent

DEFAULT_NORMALIZATION_CHUNK_SIZE = 100_000


def _component_scales(matrix: Array) -> Array:
    diagonal = jnp.real(jnp.diag(matrix))
    if bool(jnp.any(diagonal <= 0.0)):
        raise ValueError("Component normalization requires positive diagonal integrals")
    return 1.0 / jnp.sqrt(diagonal)


def _component_scales_unchecked(matrix: Array) -> tuple[Array, Array]:
    diagonal = jnp.real(jnp.diag(matrix))
    return 1.0 / jnp.sqrt(diagonal), diagonal


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


def _scaled_matrix_from_raw(raw_matrix: Array, scales: Array) -> Array:
    return scales[:, None] * raw_matrix * scales[None, :]


def _padded_mapping_chunk(
    data: Mapping[str, Array],
    start: int,
    stop: int,
    chunk_size: int,
) -> dict[str, Array]:
    """Slice one normalization chunk and pad its tail with a valid event.

    The accompanying padded integration weights are zero, so the repeated event
    does not contribute to the integral. Repeating a physical point instead of
    padding kinematic coordinates with zeros avoids evaluating resonance
    dynamics at unphysical coordinates in the final partial chunk.
    """

    count = stop - start
    if count < 1:
        raise ValueError("normalization chunks must contain at least one point")
    result: dict[str, Array] = {}
    padding = chunk_size - count
    for key, value in data.items():
        array = jnp.asarray(value)
        piece = array[start:stop]
        if padding:
            filler = jnp.broadcast_to(
                piece[:1],
                (padding,) + piece.shape[1:],
            )
            piece = jnp.concatenate((piece, filler), axis=0)
        result[key] = piece
    return result


def _padded_vector_chunk(
    values: Array,
    start: int,
    stop: int,
    chunk_size: int,
    *,
    padding_value: float,
) -> Array:
    piece = jnp.asarray(values)[start:stop]
    padding = chunk_size - (stop - start)
    if padding:
        filler = jnp.full((padding,), padding_value, dtype=piece.dtype)
        piece = jnp.concatenate((piece, filler), axis=0)
    return piece


def _compact_normalization_chunk_kernel(
    components: tuple[AmplitudeComponent, ...],
    *,
    has_efficiency: bool,
):
    """Compile one fixed-size normalization chunk.

    Returning matrix *sums* rather than means lets the caller combine an
    arbitrary number of chunks and divide only once by the true total number of
    quadrature points.  The physics is therefore identical to evaluating the
    full normalization array in a single call.
    """

    def kernel(normalization_data, weights, efficiency):
        prepared_norm = _prepare_component_data(components, normalization_data)
        raw_norm = jnp.stack(
            [jnp.asarray(c.function(prepared_norm, None)) for c in components],
            axis=1,
        )
        raw_sum = jnp.einsum(
            "n,ni,nj->ij",
            weights,
            jnp.conj(raw_norm),
            raw_norm,
        )
        if has_efficiency:
            efficient_sum = jnp.einsum(
                "n,ni,nj->ij",
                weights * efficiency,
                jnp.conj(raw_norm),
                raw_norm,
            )
        else:
            efficient_sum = raw_sum
        return raw_sum, efficient_sum

    return jax.jit(kernel)


def _compact_normalization_kernel(
    components: tuple[AmplitudeComponent, ...],
    *,
    normalize_components: bool,
    has_efficiency: bool,
    chunk_size: int = DEFAULT_NORMALIZATION_CHUNK_SIZE,
):
    """Build a chunked normalization program for coefficient-only fits.

    XLA compilation time grows strongly with the static normalization-array
    shape.  Evaluating a million-point grid in fixed-size chunks keeps the
    compiled graph at the much smaller chunk shape while preserving the exact
    weighted matrix integral.  All chunks reuse the same executable.
    """

    if chunk_size < 1:
        raise ValueError("normalization chunk_size must be positive")

    chunk_kernel = _compact_normalization_chunk_kernel(
        components,
        has_efficiency=has_efficiency,
    )

    def kernel(normalization_data, weights, efficiency):
        weights_array = jnp.asarray(weights)
        n_points = int(weights_array.shape[0])
        if n_points < 1:
            raise ValueError("normalization sample must contain at least one point")
        active_chunk_size = min(int(chunk_size), n_points)

        raw_parts = []
        efficient_parts = []
        for start in range(0, n_points, active_chunk_size):
            stop = min(start + active_chunk_size, n_points)
            data_chunk = _padded_mapping_chunk(
                normalization_data,
                start,
                stop,
                active_chunk_size,
            )
            weight_chunk = _padded_vector_chunk(
                weights_array,
                start,
                stop,
                active_chunk_size,
                padding_value=0.0,
            )
            efficiency_chunk = _padded_vector_chunk(
                efficiency,
                start,
                stop,
                active_chunk_size,
                padding_value=1.0,
            )
            raw_part, efficient_part = chunk_kernel(
                data_chunk,
                weight_chunk,
                efficiency_chunk,
            )
            raw_parts.append(raw_part)
            efficient_parts.append(efficient_part)

        raw_matrix = jnp.sum(jnp.stack(raw_parts, axis=0), axis=0) / n_points
        efficient_matrix = (
            jnp.sum(jnp.stack(efficient_parts, axis=0), axis=0) / n_points
        )

        if normalize_components:
            scales, diagonal = _component_scales_unchecked(raw_matrix)
            fixed_matrix = _scaled_matrix_from_raw(efficient_matrix, scales)
        else:
            scales = jnp.ones(
                (raw_matrix.shape[0],),
                dtype=jnp.real(raw_matrix).dtype,
            )
            diagonal = jnp.real(jnp.diag(raw_matrix))
            fixed_matrix = efficient_matrix

        return fixed_matrix, diagonal, scales

    # Expose the reusable compiled chunk function for diagnostics/benchmarks.
    kernel.chunk_kernel = chunk_kernel
    kernel.chunk_size = int(chunk_size)
    return kernel


def _compact_data_kernel(
    components: tuple[AmplitudeComponent, ...],
    *,
    normalize_components: bool,
):
    """Build a data-only kernel for datasets with fixed normalization."""

    def kernel(data, scales):
        prepared_data = _prepare_component_data(components, data)
        raw_data = jnp.stack(
            [jnp.asarray(c.function(prepared_data, None)) for c in components],
            axis=1,
        )
        return raw_data * scales if normalize_components else raw_data

    return jax.jit(kernel)


def _compact_prepare_kernel(
    components: tuple[AmplitudeComponent, ...],
    *,
    normalize_components: bool,
    has_efficiency: bool,
    normalization_chunk_size: int = DEFAULT_NORMALIZATION_CHUNK_SIZE,
):
    """Compose normalization- and data-side coefficient-only programs."""

    normalization_kernel = _compact_normalization_kernel(
        components,
        normalize_components=normalize_components,
        has_efficiency=has_efficiency,
        chunk_size=normalization_chunk_size,
    )
    data_kernel = _compact_data_kernel(
        components,
        normalize_components=normalize_components,
    )

    def kernel(data, normalization_data, weights, efficiency):
        fixed_matrix, diagonal, scales = normalization_kernel(
            normalization_data,
            weights,
            efficiency,
        )
        data_components = data_kernel(data, scales)
        return data_components, fixed_matrix, diagonal, scales

    kernel.normalization_kernel = normalization_kernel
    kernel.data_kernel = data_kernel
    return kernel


@dataclass(frozen=True)
class PreparedAmplitudeCache:
    """Pre-evaluated amplitude components and normalization matrix.

    The coefficient-only normalization is evaluated in fixed-size chunks to
    bound XLA compilation cost.  Its tiny per-component scales and fixed
    normalization matrix can then be reused by the parent ``DecayModel`` so
    later datasets only need the data-side amplitude evaluation.
    """

    components: tuple[AmplitudeComponent, ...]
    data: Mapping[str, Array] | None
    normalization_data: Mapping[str, Array] | None
    normalization_weights: Array
    parameters: tuple[Parameter, ...]
    data_components: Array
    normalization_components: Array | None
    normalization_matrix_fixed: Array
    efficiency_normalization: Array | None = None
    normalize_components: bool = True
    component_scales: Array | None = None

    @staticmethod
    def build_compact_prepare_kernel(
        components: Sequence[AmplitudeComponent],
        *,
        normalize_components: bool,
        has_efficiency: bool,
        normalization_chunk_size: int = DEFAULT_NORMALIZATION_CHUNK_SIZE,
    ):
        return _compact_prepare_kernel(
            tuple(components),
            normalize_components=bool(normalize_components),
            has_efficiency=bool(has_efficiency),
            normalization_chunk_size=int(normalization_chunk_size),
        )

    @staticmethod
    def build_compact_data_kernel(
        components: Sequence[AmplitudeComponent],
        *,
        normalize_components: bool,
    ):
        return _compact_data_kernel(
            tuple(components),
            normalize_components=bool(normalize_components),
        )

    @classmethod
    def prepare_from_fixed_normalization(
        cls,
        components: Sequence[AmplitudeComponent],
        *,
        data: Mapping[str, Array],
        normalization_weights: Array,
        parameters: Sequence[Parameter],
        normalization_matrix_fixed: Array,
        component_scales: Array,
        normalize_components: bool,
        compact_data_kernel=None,
    ) -> "PreparedAmplitudeCache":
        """Prepare only dataset amplitudes using an existing model normalization."""

        components = tuple(components)
        parameters = tuple(parameters)
        if any(
            parameter.kind is ParameterKind.DYNAMICS and not parameter.fixed
            for parameter in parameters
        ):
            raise ValueError("fixed-normalization reuse requires coefficient-only dynamics")
        scales = jnp.asarray(component_scales)
        kernel = compact_data_kernel
        if kernel is None:
            kernel = cls.build_compact_data_kernel(
                components,
                normalize_components=normalize_components,
            )
        data_components = kernel(data, scales)
        return cls(
            components=components,
            data=None,
            normalization_data=None,
            normalization_weights=jnp.asarray(normalization_weights),
            parameters=parameters,
            data_components=data_components,
            normalization_components=None,
            normalization_matrix_fixed=jnp.asarray(normalization_matrix_fixed),
            efficiency_normalization=None,
            normalize_components=normalize_components,
            component_scales=scales,
        )

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
        compact_prepare_kernel=None,
        normalization_chunk_size: int = DEFAULT_NORMALIZATION_CHUNK_SIZE,
    ) -> "PreparedAmplitudeCache":
        components = tuple(components)
        parameters = tuple(parameters)
        if not components:
            raise ValueError("At least one amplitude component is required")

        weights = jnp.asarray(normalization_weights)
        has_floating_dynamics = any(
            parameter.kind is ParameterKind.DYNAMICS and not parameter.fixed
            for parameter in parameters
        )

        if not has_floating_dynamics:
            efficiency = (
                jnp.asarray(efficiency_normalization)
                if efficiency_normalization is not None
                else jnp.ones_like(weights)
            )
            kernel = compact_prepare_kernel
            if kernel is None:
                kernel = cls.build_compact_prepare_kernel(
                    components,
                    normalize_components=normalize_components,
                    has_efficiency=efficiency_normalization is not None,
                    normalization_chunk_size=normalization_chunk_size,
                )
            data_components, fixed_matrix, diagonal, scales = kernel(
                data,
                normalization_data,
                weights,
                efficiency,
            )
            if bool(jnp.any(diagonal <= 0.0)):
                raise ValueError(
                    "Component normalization requires positive diagonal integrals"
                )
            return cls(
                components=components,
                data=None,
                normalization_data=None,
                normalization_weights=weights,
                parameters=parameters,
                data_components=data_components,
                normalization_components=None,
                normalization_matrix_fixed=fixed_matrix,
                efficiency_normalization=efficiency_normalization,
                normalize_components=normalize_components,
                component_scales=scales,
            )

        prepared_data = _prepare_component_data(components, data)
        prepared_norm = _prepare_component_data(components, normalization_data)
        raw_data = jnp.stack(
            [jnp.asarray(c.function(prepared_data, None)) for c in components], axis=1
        )
        raw_norm = jnp.stack(
            [jnp.asarray(c.function(prepared_norm, None)) for c in components], axis=1
        )

        raw_component_matrix = normalization_matrix(raw_norm, weights, None)
        if normalize_components:
            scales = _component_scales(raw_component_matrix)
            data_components = raw_data * scales
            norm_components = raw_norm * scales
        else:
            scales = jnp.ones(
                (raw_norm.shape[1],), dtype=jnp.real(raw_component_matrix).dtype
            )
            data_components = raw_data
            norm_components = raw_norm

        if efficiency_normalization is not None:
            fixed_matrix = normalization_matrix(
                norm_components,
                weights,
                efficiency_normalization,
            )
        elif normalize_components:
            fixed_matrix = _scaled_matrix_from_raw(raw_component_matrix, scales)
        else:
            fixed_matrix = raw_component_matrix

        return cls(
            components=components,
            data=prepared_data,
            normalization_data=prepared_norm,
            normalization_weights=weights,
            parameters=parameters,
            data_components=data_components,
            normalization_components=norm_components,
            normalization_matrix_fixed=fixed_matrix,
            efficiency_normalization=efficiency_normalization,
            normalize_components=normalize_components,
            component_scales=scales,
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

    @property
    def is_compact(self) -> bool:
        return not self.floating_dynamic_owners

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
        self,
        fit_values: Mapping[str, object],
    ) -> tuple[Array, Array | None]:
        owners = self.floating_dynamic_owners
        if not owners:
            return self.data_components, None
        if self.data is None or self.normalization_data is None:
            raise RuntimeError("Dynamic cache is missing prepared event data")
        if self.normalization_components is None:
            raise RuntimeError("Dynamic cache is missing normalization components")

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

    def _matrix_with_dynamic_blocks(self, norm_components: Array | None) -> Array:
        owners = self.floating_dynamic_owners
        if not owners:
            return self.normalization_matrix_fixed
        if norm_components is None:
            raise RuntimeError("Dynamic normalization components are required")

        dynamic_indices = tuple(
            i for i, component in enumerate(self.components) if component.name in owners
        )
        index = jnp.asarray(dynamic_indices, dtype=jnp.int32)
        dynamic = norm_components[:, index]
        weights = self._pdf_weights()

        rows = jnp.einsum(
            "n,nd,nj->dj",
            weights,
            jnp.conj(dynamic),
            norm_components,
        ) / norm_components.shape[0]

        matrix = self.normalization_matrix_fixed
        matrix = matrix.at[index, :].set(rows)
        matrix = matrix.at[:, index].set(jnp.conj(rows).T)
        diagonal = jnp.real(rows[jnp.arange(index.shape[0]), index])
        matrix = matrix.at[index, index].set(diagonal)
        return matrix

    def evaluate(self, fit_values: Mapping[str, object]) -> tuple[Array, Array]:
        coefficients = self.coefficient_vector(fit_values)
        data_components, norm_components = self._evaluate_components(fit_values)
        amplitude = data_components @ coefficients
        intensity = jnp.abs(amplitude) ** 2
        normalization = matrix_normalization(
            coefficients,
            self._matrix_with_dynamic_blocks(norm_components),
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
            coefficients,
            self._matrix_with_dynamic_blocks(norm_components),
        )

    def normalization_matrix(self, fit_values: Mapping[str, object]) -> Array:
        _, norm_components = self._evaluate_components(fit_values)
        return self._matrix_with_dynamic_blocks(norm_components)

    def fit_fractions(self, fit_values: Mapping[str, object]) -> Array:
        return matrix_fit_fractions(
            self.coefficient_vector(fit_values),
            self.normalization_matrix(fit_values),
        )

    def interference_fractions(self, fit_values: Mapping[str, object]) -> Array:
        return matrix_interference_fractions(
            self.coefficient_vector(fit_values),
            self.normalization_matrix(fit_values),
        )
