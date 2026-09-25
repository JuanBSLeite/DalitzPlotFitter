"""Cached coherent amplitude evaluation for repeated likelihood calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.fit import Parameter, ParameterKind
from dalitzplotfitter.integration import matrix_normalization, normalization_matrix
from dalitzplotfitter.observables import fit_fractions as matrix_fit_fractions
from dalitzplotfitter.observables import (
    interference_fractions as matrix_interference_fractions,
)

from .components import AmplitudeComponent, coefficient_value

DEFAULT_NORMALIZATION_CHUNK_SIZE = 100_000
DEFAULT_DYNAMICS_MICROBATCH_PARALLELISM = 1

# Component-function type names already warned about below, so a chunked
# normalization pass (or a loop of toy fits) does not reprint the same
# diagnostic once per chunk/toy.
_MISSING_COMPACTION_WARNED: set[str] = set()

# (order-dependent component names, requested chunk size, microbatch size)
# combinations already reported, so a multi-toy loop preparing the same model
# configuration repeatedly does not reprint the same diagnostic per toy.
_CHUNK_CAP_WARNED: set[tuple[tuple[str, ...], int, int]] = set()


@partial(jax.jit, donate_argnums=(0,))
def _store_normalization_chunk(buffers, chunk, index):
    """Fill owned cache buffers without retaining a second full-grid copy."""
    return jax.tree_util.tree_map(
        lambda buffer, values: jax.lax.dynamic_update_index_in_dim(
            buffer, values, index, axis=0
        ),
        buffers,
        chunk,
    )


def _component_normalization_mask(
    components: Sequence[AmplitudeComponent],
    normalize_components: bool,
) -> tuple[bool, ...]:
    return tuple(
        _normalize_component(component, normalize_components)
        for component in components
    )


def _normalize_component(
    component: AmplitudeComponent,
    normalize_components: bool,
) -> bool:
    if component.normalize_component is None:
        return bool(normalize_components)
    return bool(component.normalize_component)


def _component_scales(matrix: Array, normalization_mask: Array) -> Array:
    diagonal = jnp.real(jnp.diag(matrix))
    if bool(jnp.any(diagonal <= 0.0)):
        raise ValueError("Component normalization requires positive diagonal integrals")
    normalized_scales = 1.0 / jnp.sqrt(diagonal)
    return jnp.where(normalization_mask, normalized_scales, 1.0)


def _component_scales_unchecked(
    matrix: Array,
    normalization_mask: Array,
) -> tuple[Array, Array]:
    diagonal = jnp.real(jnp.diag(matrix))
    normalized_scales = 1.0 / jnp.sqrt(diagonal)
    return jnp.where(normalization_mask, normalized_scales, 1.0), diagonal


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


def _minimal_component_input(
    components: Sequence[AmplitudeComponent],
    data: Mapping[str, Array],
) -> Mapping[str, Array]:
    """Drop four-momenta when all selected components support invariants."""

    components = tuple(components)
    invariant_keys = ("s12", "s13", "s23")
    if (
        components
        and all(
            bool(getattr(component.function, "supports_invariant_input", False))
            for component in components
        )
        and all(key in data for key in invariant_keys)
    ):
        return {key: data[key] for key in invariant_keys}
    return dict(data)


def _compact_prepared_component_data(
    components: Sequence[AmplitudeComponent],
    prepared: Mapping[str, Array],
) -> Mapping[str, Array]:
    """Retain the union of component-specific prepared arrays when possible."""

    components = tuple(components)
    if not components:
        return {}
    compacted = []
    for component in components:
        compact = getattr(component.function, "compact_prepared_data", None)
        if compact is None:
            type_name = type(component.function).__name__
            if type_name not in _MISSING_COMPACTION_WARNED:
                _MISSING_COMPACTION_WARNED.add(type_name)
                print(
                    "WARNING DalitzPlotFitter normalization: floating component "
                    f"{component.name!r} ({type_name}) has no compact_prepared_data "
                    "method, so the full shared prepared mapping is retained for "
                    "every floating component in this model instead of only what "
                    "each one needs; see 'compact_prepared_data must be defined for "
                    "every floating component type' in docs/performance.md."
                )
            return prepared
        compacted.append(compact(prepared))

    result = {}
    for mapping in compacted:
        result.update(mapping)
    return result


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


# A normalization_chunk_size chosen for XLA compilation/geometry-storage
# amortization (commonly 1e5-1e6) is far larger than the AD reverse-pass can
# hold live for several floating-dynamics lineshapes at once. Every
# floating-dynamics chunk is therefore re-split into configurable
# microbatches, each wrapped in its own `jax.checkpoint`, so peak memory for
# one macro-chunk stays bounded by one microbatch regardless of the total grid
# size. DecayModel exposes this default as `dynamics_microbatch_size`.
DEFAULT_DYNAMICS_MICROBATCH_SIZE = 20_000


def _balanced_block_size(point_count: int, maximum_size: int) -> int:
    """Choose an almost-even static block size no larger than ``maximum_size``.

    XLA scans require every block to have the same shape.  Treating the public
    setting as that exact shape can waste nearly one complete block when the
    sample is just over a multiple of it.  Instead, first choose the minimum
    number of blocks allowed by the memory cap, then distribute the points as
    evenly as a single static shape permits.
    """
    block_count = (point_count + maximum_size - 1) // maximum_size
    return (point_count + block_count - 1) // block_count


def _has_order_dependent_preparation(
    components: Sequence[AmplitudeComponent],
) -> bool:
    return any(
        getattr(
            getattr(component.function, "lineshape", None),
            "prepared_mass_is_order_dependent",
            False,
        )
        for component in components
    )


def _repeat_first_padded(array: Array, micro_size: int) -> Array:
    """Reshape ``(n, ...)`` into ``(n // micro_size, micro_size, ...)``.

    The leading axis is padded to a multiple of ``micro_size`` by repeating
    the first row, matching ``_padded_mapping_chunk``'s convention of padding
    with a valid physical point so dynamics are never evaluated at
    unphysical coordinates.
    """
    n = array.shape[0]
    remainder = n % micro_size
    if remainder:
        pad_count = micro_size - remainder
        filler = jnp.broadcast_to(array[:1], (pad_count,) + array.shape[1:])
        array = jnp.concatenate((array, filler), axis=0)
    return array.reshape((-1, micro_size) + array.shape[1:])


def _value_padded(array: Array, micro_size: int, padding_value: float) -> Array:
    """Reshape ``(n, ...)`` into ``(n // micro_size, micro_size, ...)``.

    The leading axis is padded to a multiple of ``micro_size`` with a
    constant (zero weight, unit efficiency), matching ``_padded_vector_chunk``.
    """
    n = array.shape[0]
    remainder = n % micro_size
    if remainder:
        pad_count = micro_size - remainder
        filler = jnp.full(
            (pad_count,) + array.shape[1:], padding_value, dtype=array.dtype
        )
        array = jnp.concatenate((array, filler), axis=0)
    return array.reshape((-1, micro_size) + array.shape[1:])


def _repeat_first_axis(array: Array, target_size: int) -> Array:
    """Pad the leading axis by repeating its first entry."""
    padding = target_size - array.shape[0]
    if padding <= 0:
        return array
    filler = jnp.broadcast_to(array[:1], (padding,) + array.shape[1:])
    return jnp.concatenate((array, filler), axis=0)


def _constant_pad_axis(
    array: Array,
    target_size: int,
    padding_value: float,
) -> Array:
    """Pad the leading axis with a constant value."""
    padding = target_size - array.shape[0]
    if padding <= 0:
        return array
    filler = jnp.full(
        (padding,) + array.shape[1:],
        padding_value,
        dtype=array.dtype,
    )
    return jnp.concatenate((array, filler), axis=0)


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

    normalization_flags = _component_normalization_mask(
        components,
        normalize_components,
    )
    normalization_mask = jnp.asarray(normalization_flags)
    has_component_normalization = any(normalization_flags)
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

        if has_component_normalization:
            scales, diagonal = _component_scales_unchecked(
                raw_matrix,
                normalization_mask,
            )
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
        return raw_data * scales

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

    ``prepare()`` bakes the fixed/floating status of every DYNAMICS parameter
    into the evaluation path once, from the ``parameters`` given at prepare
    time. Advanced callers assembling ``Minimizer`` directly (bypassing
    ``DecayModel``/``FitSession``) must pass that *same* parameter list to
    ``Minimizer`` — passing a different list that disagrees on which DYNAMICS
    parameters are fixed produces no error but silently drops that
    parameter's gradient to exactly zero. Call ``check_parameters`` to guard
    against this when the two lists are not obviously the same object.
    """

    components: tuple[AmplitudeComponent, ...]
    data: Mapping[str, Array] | None
    normalization_data: Mapping[str, Array] | None
    normalization_weights: Array
    parameters: tuple[Parameter, ...]
    data_components: Array
    normalization_components: tuple[Array, ...] | Array | None
    normalization_matrix_fixed: Array
    efficiency_normalization: Array | None = None
    normalize_components: bool = True
    component_scales: Array | None = None
    fixed_component_indices: tuple[int, ...] | None = None
    dynamic_component_indices: tuple[int, ...] | None = None
    normalization_chunks: tuple | None = None
    normalization_chunk_size: int | None = None
    dynamics_microbatch_size: int = DEFAULT_DYNAMICS_MICROBATCH_SIZE
    dynamics_microbatch_parallelism: int = DEFAULT_DYNAMICS_MICROBATCH_PARALLELISM
    effective_normalization_chunk_size: int | None = None
    effective_dynamics_microbatch_size: int | None = None
    effective_dynamics_microbatch_parallelism: int | None = None

    @property
    def normalization_padding_points(self) -> int:
        """Number of zero-weight positions evaluated by dynamic normalization."""
        if self.normalization_chunks is None:
            return 0
        leaves = jax.tree_util.tree_leaves(self.normalization_chunks)
        if not leaves:
            return 0
        macro_count, macro_size = map(int, leaves[0].shape[:2])
        micro_size = self.effective_dynamics_microbatch_size or macro_size
        micro_count = (macro_size + micro_size - 1) // micro_size
        processed = macro_count * micro_count * micro_size
        return processed - int(self.normalization_weights.size)

    @property
    def normalization_padding_fraction(self) -> float:
        """Fraction of dynamic-normalization work spent on padded positions."""
        return self.normalization_padding_points / int(self.normalization_weights.size)

    @staticmethod
    def build_compact_prepare_kernel(
        components: Sequence[AmplitudeComponent],
        *,
        normalize_components: bool,
        has_efficiency: bool,
        normalization_chunk_size: int = DEFAULT_NORMALIZATION_CHUNK_SIZE,
    ):
        """Build the reusable chunked coefficient-only prepare kernel."""
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
        """Build the reusable data-only kernel for fixed normalization."""
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
    ) -> PreparedAmplitudeCache:
        """Prepare only dataset amplitudes using an existing model normalization."""

        components = tuple(components)
        parameters = tuple(parameters)
        if any(
            parameter.kind is ParameterKind.DYNAMICS and not parameter.fixed
            for parameter in parameters
        ):
            raise ValueError(
                "fixed-normalization reuse requires coefficient-only dynamics"
            )
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
        dynamics_microbatch_size: int = DEFAULT_DYNAMICS_MICROBATCH_SIZE,
        dynamics_microbatch_parallelism: int = DEFAULT_DYNAMICS_MICROBATCH_PARALLELISM,
    ) -> PreparedAmplitudeCache:
        """Evaluate components and the normalization matrix once, from scratch.

        Coefficient-only fits (no floating DYNAMICS parameter) take the fast
        compact path and never re-evaluate lineshapes afterwards; fits with a
        floating DYNAMICS parameter instead partition fixed vs. dynamic
        components so only the dynamic block is re-evaluated per step.
        """
        components = tuple(components)
        parameters = tuple(parameters)
        if not components:
            raise ValueError("At least one amplitude component is required")

        if normalization_chunk_size < 1:
            raise ValueError("normalization_chunk_size must be positive")
        if (
            isinstance(dynamics_microbatch_size, bool)
            or not isinstance(dynamics_microbatch_size, int)
            or dynamics_microbatch_size < 1
        ):
            raise ValueError("dynamics_microbatch_size must be a positive integer")
        if (
            isinstance(dynamics_microbatch_parallelism, bool)
            or not isinstance(dynamics_microbatch_parallelism, int)
            or dynamics_microbatch_parallelism < 1
        ):
            raise ValueError(
                "dynamics_microbatch_parallelism must be a positive integer"
            )
        weights = jnp.asarray(normalization_weights)
        if weights.size < 1:
            raise ValueError("normalization sample must contain at least one point")
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
                dynamics_microbatch_parallelism=dynamics_microbatch_parallelism,
            )

        # Floating-dynamics fits (notably QMI) must keep the normalization
        # re-evaluable, but they must not materialize an N x Ncomponents
        # complex matrix.  For the B->3pi adaptive grid this matrix is about
        # 168 MiB for six components and was the dominant contiguous GPU
        # allocation.  Partition first, keep fixed normalization amplitudes as
        # separate columns, and build only the tiny matrix blocks.
        floating_owners = frozenset(
            parameter.owner
            for parameter in parameters
            if (
                parameter.kind is ParameterKind.DYNAMICS
                and not parameter.fixed
                and parameter.owner is not None
            )
        )
        dynamic_indices = tuple(
            index
            for index, component in enumerate(components)
            if component.name in floating_owners
        )
        fixed_indices = tuple(
            index for index in range(len(components)) if index not in dynamic_indices
        )
        dynamic_components = tuple(components[index] for index in dynamic_indices)
        dynamic_chunk_limit = min(int(normalization_chunk_size), int(weights.size))
        order_dependent_preparation = _has_order_dependent_preparation(
            dynamic_components
        )
        if order_dependent_preparation:
            # QMI's sort indices belong to the exact block passed to
            # prepare_mass. Prepare smaller blocks up front rather than
            # slicing that state later in the inner AD microbatch scan.
            if int(dynamics_microbatch_size) < int(normalization_chunk_size):
                order_dependent_names = tuple(
                    component.name
                    for component in dynamic_components
                    if getattr(
                        getattr(component.function, "lineshape", None),
                        "prepared_mass_is_order_dependent",
                        False,
                    )
                )
                cap_key = (
                    order_dependent_names,
                    int(normalization_chunk_size),
                    int(dynamics_microbatch_size),
                )
                if cap_key not in _CHUNK_CAP_WARNED:
                    _CHUNK_CAP_WARNED.add(cap_key)
                    print(
                        "INFO DalitzPlotFitter normalization: floating component(s) "
                        f"{order_dependent_names} use order-dependent prepared state "
                        "(e.g. QMI's sort order/interval boundaries), so the effective "
                        "normalization chunk size is capped at "
                        f"dynamics_microbatch_size={int(dynamics_microbatch_size)} "
                        "instead of the requested normalization_chunk_size="
                        f"{int(normalization_chunk_size)}; see 'normalization_chunk_size "
                        "is silently capped by dynamics_microbatch_size...' in "
                        "docs/performance.md."
                    )
            dynamic_chunk_limit = min(
                dynamic_chunk_limit,
                dynamics_microbatch_size,
            )
        needs_inner_microbatch = (
            not order_dependent_preparation
            and weights.size > dynamics_microbatch_size
        )
        if weights.size > dynamic_chunk_limit or needs_inner_microbatch:
            dynamic_chunk_size = _balanced_block_size(
                int(weights.size), dynamic_chunk_limit
            )
            return cls._prepare_chunked_dynamics(
                components,
                data,
                normalization_data,
                weights,
                parameters,
                efficiency_normalization,
                normalize_components,
                fixed_indices,
                dynamic_indices,
                dynamic_chunk_size,
                dynamics_microbatch_size,
                int(normalization_chunk_size),
                dynamics_microbatch_parallelism,
            )

        fixed_components = tuple(components[index] for index in fixed_indices)

        minimal_data = _minimal_component_input(fixed_components, data)
        minimal_norm = _minimal_component_input(fixed_components, normalization_data)
        raw_fixed_data = tuple(
            jnp.asarray(component.function(minimal_data, None))
            for component in fixed_components
        )
        raw_fixed_norm = tuple(
            jnp.asarray(component.function(minimal_norm, None))
            for component in fixed_components
        )

        normalization_flags = _component_normalization_mask(
            components,
            normalize_components,
        )
        real_dtype = jnp.result_type(jnp.asarray(weights).dtype, jnp.float32)
        complex_dtype = jnp.result_type(real_dtype, jnp.complex64)
        scales = jnp.ones((len(components),), dtype=real_dtype)

        # Component normalization uses the physical quadrature weights only;
        # efficiency enters the PDF matrix afterwards.
        for local_index, component_index in enumerate(fixed_indices):
            if normalization_flags[component_index]:
                diagonal = jnp.real(
                    jnp.mean(weights * jnp.abs(raw_fixed_norm[local_index]) ** 2)
                )
                if bool(diagonal <= 0.0):
                    raise ValueError(
                        "Component normalization requires positive diagonal integrals"
                    )
                scales = scales.at[component_index].set(1.0 / jnp.sqrt(diagonal))

        pdf_weights = weights
        if efficiency_normalization is not None:
            pdf_weights = pdf_weights * jnp.asarray(efficiency_normalization)

        fixed_matrix = jnp.zeros(
            (len(components), len(components)),
            dtype=complex_dtype,
        )
        n_points = int(weights.shape[0])
        for left_local, left_index in enumerate(fixed_indices):
            left_values = raw_fixed_norm[left_local] * scales[left_index]
            for right_local in range(left_local, len(fixed_indices)):
                right_index = fixed_indices[right_local]
                right_values = raw_fixed_norm[right_local] * scales[right_index]
                entry = (
                    jnp.einsum(
                        "n,n,n->",
                        pdf_weights,
                        jnp.conj(left_values),
                        right_values,
                    )
                    / n_points
                )
                fixed_matrix = fixed_matrix.at[left_index, right_index].set(entry)
                if right_index != left_index:
                    fixed_matrix = fixed_matrix.at[right_index, left_index].set(
                        jnp.conj(entry)
                    )

        if fixed_indices:
            data_components = jnp.stack(
                [
                    raw_fixed_data[local] * scales[index]
                    for local, index in enumerate(fixed_indices)
                ],
                axis=1,
            )
        else:
            n_data = int(next(iter(data.values())).shape[0])
            data_components = jnp.zeros(
                (n_data, 0),
                dtype=complex_dtype,
            )

        retained_data = _prepare_component_data(
            dynamic_components,
            _minimal_component_input(dynamic_components, data),
        )
        retained_norm = _prepare_component_data(
            dynamic_components,
            _minimal_component_input(dynamic_components, normalization_data),
        )
        retained_data = _compact_prepared_component_data(
            dynamic_components,
            retained_data,
        )
        retained_norm = _compact_prepared_component_data(
            dynamic_components,
            retained_norm,
        )

        return cls(
            components=components,
            data=retained_data,
            normalization_data=retained_norm,
            normalization_weights=weights,
            parameters=parameters,
            data_components=data_components,
            # Keep each fixed normalization amplitude in its own allocation.
            # This avoids the large contiguous jnp.stack allocation.
            normalization_components=raw_fixed_norm,
            normalization_matrix_fixed=fixed_matrix,
            efficiency_normalization=efficiency_normalization,
            normalize_components=normalize_components,
            component_scales=scales,
            fixed_component_indices=fixed_indices,
            dynamic_component_indices=dynamic_indices,
            dynamics_microbatch_size=dynamics_microbatch_size,
            dynamics_microbatch_parallelism=dynamics_microbatch_parallelism,
        )

    @classmethod
    def _prepare_chunked_dynamics(
        cls,
        components,
        data,
        normalization_data,
        weights,
        parameters,
        efficiency,
        normalize,
        fixed_indices,
        dynamic_indices,
        chunk_size,
        dynamics_microbatch_size,
        requested_chunk_size,
        dynamics_microbatch_parallelism,
    ):
        """Prepare geometry independently in each normalization block.

        In particular, QMI sorting indices and interval boundaries are local
        to a block; slicing a full-grid prepared QMI mapping is not valid.
        Fixed amplitudes are evaluated only here, never during minimization.
        """
        fixed = tuple(components[i] for i in fixed_indices)
        dynamic = tuple(components[i] for i in dynamic_indices)
        n_points = weights.shape[0]
        can_microbatch = not _has_order_dependent_preparation(dynamic)
        effective_microbatch_size = (
            _balanced_block_size(
                int(chunk_size), min(int(dynamics_microbatch_size), int(chunk_size))
            )
            if can_microbatch
            else int(chunk_size)
        )
        microbatch_count = (
            (int(chunk_size) + effective_microbatch_size - 1)
            // effective_microbatch_size
        )
        effective_parallelism = (
            min(int(dynamics_microbatch_parallelism), microbatch_count)
            if can_microbatch
            else 1
        )
        efficiency_array = (
            jnp.ones_like(weights) if efficiency is None else jnp.asarray(efficiency)
        )
        norm_input = _minimal_component_input(components, normalization_data)

        @jax.jit
        def prepare_chunk(events, w, eff):
            prepared = _prepare_component_data(components, events)
            columns = tuple(jnp.asarray(c.function(prepared, None)) for c in fixed)
            if fixed:
                values = jnp.stack(columns, axis=1)
                bare = jnp.einsum("n,ni,nj->ij", w, values.conj(), values)
                accepted = jnp.einsum("n,ni,nj->ij", w * eff, values.conj(), values)
            else:
                dtype = jnp.result_type(w.dtype, jnp.complex64)
                bare = accepted = jnp.zeros((0, 0), dtype=dtype)
            geometry = _compact_prepared_component_data(dynamic, prepared)
            return (geometry, w, eff, columns), bare, accepted

        chunk_arrays = None
        n_chunks = (n_points + chunk_size - 1) // chunk_size
        bare_sum = accepted_sum = None
        for start in range(0, n_points, chunk_size):
            stop = min(start + chunk_size, n_points)
            chunk, bare, accepted = prepare_chunk(
                _padded_mapping_chunk(norm_input, start, stop, chunk_size),
                _padded_vector_chunk(weights, start, stop, chunk_size, padding_value=0),
                _padded_vector_chunk(
                    efficiency_array,
                    start,
                    stop,
                    chunk_size,
                    padding_value=1,
                ),
            )
            if chunk_arrays is None:
                chunk_arrays = jax.tree_util.tree_map(
                    lambda array: jnp.zeros(
                        (n_chunks,) + array.shape, dtype=array.dtype
                    ),
                    chunk,
                )
            chunk_arrays = _store_normalization_chunk(
                chunk_arrays, chunk, start // chunk_size
            )
            bare_sum = bare if bare_sum is None else bare_sum + bare
            accepted_sum = accepted if accepted_sum is None else accepted_sum + accepted
        bare_matrix = bare_sum / n_points
        accepted_matrix = accepted_sum / n_points
        real_dtype = jnp.result_type(weights.dtype, jnp.float32)
        complex_dtype = jnp.result_type(real_dtype, jnp.complex64)
        scales = jnp.ones((len(components),), dtype=real_dtype)
        matrix = jnp.zeros((len(components), len(components)), dtype=complex_dtype)
        if fixed:
            flags = jnp.asarray(_component_normalization_mask(fixed, normalize))
            diagonal = jnp.real(jnp.diag(bare_matrix))
            if bool(jnp.any(flags & (diagonal <= 0))):
                raise ValueError(
                    "Component normalization requires positive diagonal integrals"
                )
            fixed_scales = 1 / jnp.sqrt(jnp.where(flags, diagonal, 1.0))
            indices = jnp.asarray(fixed_indices)
            scales = scales.at[indices].set(fixed_scales)
            matrix = matrix.at[indices[:, None], indices[None, :]].set(
                _scaled_matrix_from_raw(accepted_matrix, fixed_scales)
            )
        prepared_data = _prepare_component_data(
            components,
            _minimal_component_input(components, data),
        )
        if fixed:
            values = (
                jnp.stack(
                    [jnp.asarray(c.function(prepared_data, None)) for c in fixed],
                    axis=1,
                )
                * scales[jnp.asarray(fixed_indices)]
            )
        else:
            values = jnp.zeros(
                (next(iter(data.values())).shape[0], 0),
                dtype=complex_dtype,
            )
        return cls(
            components=components,
            parameters=parameters,
            data=_compact_prepared_component_data(dynamic, prepared_data),
            normalization_data=None,
            normalization_weights=weights,
            data_components=values,
            normalization_components=None,
            normalization_matrix_fixed=matrix,
            component_scales=scales,
            efficiency_normalization=efficiency,
            normalize_components=normalize,
            fixed_component_indices=fixed_indices,
            dynamic_component_indices=dynamic_indices,
            normalization_chunks=chunk_arrays,
            normalization_chunk_size=requested_chunk_size,
            dynamics_microbatch_size=dynamics_microbatch_size,
            dynamics_microbatch_parallelism=dynamics_microbatch_parallelism,
            effective_normalization_chunk_size=chunk_size,
            effective_dynamics_microbatch_size=effective_microbatch_size,
            effective_dynamics_microbatch_parallelism=effective_parallelism,
        )

    def _chunked_dynamic_normalization(self, fit_values):
        """Accumulate raw integrals, then apply global component scales.

        Checkpoint the scan body so reverse AD recomputes one block instead of
        retaining event-sized dynamics for every block. The carry consists only
        of small matrix blocks and bare component diagonals. The denominator is
        the original sample size, not the padded size or the number of blocks.

        Each macro-chunk (sized by ``normalization_chunk_size``, chosen for
        XLA compilation/geometry-storage amortization) is itself re-split into
        ``dynamics_microbatch_size``-sized microbatches via a nested,
        checkpointed scan, so the reverse-AD pass through several floating
        dynamics lineshapes never has to hold more than one microbatch's
        forward residuals live at a time -- independent of the total grid
        size.
        """
        fixed, dynamic = self._component_partitions()
        n_dynamic = len(dynamic)
        dtype = self.normalization_matrix_fixed.dtype
        real_dtype = self.component_scales.dtype
        mappings = tuple(
            self._dynamic_parameter_mapping(self.components[i].name, fit_values)
            for i in dynamic
        )
        # QMI's prepared data (see `QMI.prepare_mass`) encodes a sort order and
        # interval boundaries over the *entire* block it was prepared on; a
        # microbatch reshape would desync those indices from the smaller
        # sub-blocks. Only microbatch when every dynamic component's lineshape
        # is free of such block-wide state (the default for anything that
        # doesn't declare otherwise, e.g. ordinary parametric lineshapes and
        # KMatrix's per-event response column).
        can_microbatch = not _has_order_dependent_preparation(
            tuple(self.components[i] for i in dynamic)
        )

        def evaluate_microbatch_values(micro_chunk):
            events, weights, efficiency, fixed_columns = micro_chunk
            values = jnp.stack(
                [
                    jnp.asarray(self.components[i].function(events, pars), dtype=dtype)
                    for i, pars in zip(dynamic, mappings, strict=True)
                ],
                axis=1,
            )
            pdf_weights = weights * efficiency
            diagonal = jnp.einsum("n,nd->d", weights, jnp.abs(values) ** 2)
            block = jnp.einsum(
                "n,nd,ne->de",
                pdf_weights,
                values.conj(),
                values,
            )
            if fixed:
                fixed_values = jnp.stack(fixed_columns, axis=1)
                fixed_values = fixed_values * self.component_scales[jnp.asarray(fixed)]
                cross = jnp.einsum(
                    "n,nd,nf->df",
                    pdf_weights,
                    values.conj(),
                    fixed_values,
                )
            else:
                cross = jnp.zeros((n_dynamic, 0), dtype=dtype)
            return diagonal, cross, block

        def add_microbatch_values(carry, partial):
            return tuple(
                previous + current
                for previous, current in zip(carry, partial, strict=True)
            )

        def evaluate_microbatch(carry, micro_chunk):
            return add_microbatch_values(
                carry,
                evaluate_microbatch_values(micro_chunk),
            ), None

        def accumulate(carry, chunk):
            events, weights, efficiency, fixed_columns = chunk
            if not can_microbatch:
                return evaluate_microbatch(carry, chunk)
            micro_size = self.effective_dynamics_microbatch_size
            if micro_size is None:
                micro_size = _balanced_block_size(
                    int(weights.shape[0]),
                    min(self.dynamics_microbatch_size, int(weights.shape[0])),
                )
            micro_events = jax.tree_util.tree_map(
                lambda a: _repeat_first_padded(a, micro_size),
                events,
            )
            micro_weights = _value_padded(weights, micro_size, 0.0)
            micro_efficiency = _value_padded(efficiency, micro_size, 1.0)
            micro_fixed_columns = tuple(
                _repeat_first_padded(column, micro_size) for column in fixed_columns
            )
            parallelism = self.effective_dynamics_microbatch_parallelism or 1
            if parallelism == 1:
                carry, _ = jax.lax.scan(
                    jax.checkpoint(evaluate_microbatch),
                    carry,
                    (
                        micro_events,
                        micro_weights,
                        micro_efficiency,
                        micro_fixed_columns,
                    ),
                )
            else:
                micro_count = micro_weights.shape[0]
                group_count = (micro_count + parallelism - 1) // parallelism
                padded_count = group_count * parallelism
                grouped_events = jax.tree_util.tree_map(
                    lambda array: _repeat_first_axis(array, padded_count).reshape(
                        (group_count, parallelism) + array.shape[1:]
                    ),
                    micro_events,
                )
                grouped_weights = _constant_pad_axis(
                    micro_weights,
                    padded_count,
                    0.0,
                ).reshape(
                    (group_count, parallelism) + micro_weights.shape[1:]
                )
                grouped_efficiency = _constant_pad_axis(
                    micro_efficiency,
                    padded_count,
                    1.0,
                ).reshape(
                    (group_count, parallelism) + micro_efficiency.shape[1:]
                )
                grouped_fixed_columns = tuple(
                    _repeat_first_axis(column, padded_count).reshape(
                        (group_count, parallelism) + column.shape[1:]
                    )
                    for column in micro_fixed_columns
                )

                def evaluate_group(carry, group_chunk):
                    partials = jax.vmap(evaluate_microbatch_values)(group_chunk)
                    partial = tuple(jnp.sum(part, axis=0) for part in partials)
                    return add_microbatch_values(carry, partial), None

                carry, _ = jax.lax.scan(
                    jax.checkpoint(evaluate_group),
                    carry,
                    (
                        grouped_events,
                        grouped_weights,
                        grouped_efficiency,
                        grouped_fixed_columns,
                    ),
                )
            return carry, None

        initial = (
            jnp.zeros(n_dynamic, dtype=real_dtype),
            jnp.zeros((n_dynamic, len(fixed)), dtype=dtype),
            jnp.zeros((n_dynamic, n_dynamic), dtype=dtype),
        )
        sums, _ = jax.lax.scan(
            jax.checkpoint(accumulate),
            initial,
            self.normalization_chunks,
        )
        diagonal, cross, block = (
            part / self.normalization_weights.size for part in sums
        )
        flags = jnp.asarray(
            [
                _normalize_component(self.components[i], self.normalize_components)
                for i in dynamic
            ]
        )
        # Avoid a dormant 1/sqrt(0) derivative for an unnormalized zero component.
        scales = 1 / jnp.sqrt(jnp.where(flags, diagonal, 1.0))
        block = scales[:, None] * block * scales[None, :]
        index = jnp.asarray(dynamic)
        matrix = self.normalization_matrix_fixed.at[index[:, None], index[None, :]].set(
            block
        )
        matrix = matrix.at[index, index].set(jnp.real(jnp.diag(block)))
        if fixed:
            fixed_index = jnp.asarray(fixed)
            cross = scales[:, None] * cross
            matrix = matrix.at[index[:, None], fixed_index[None, :]].set(cross)
            matrix = matrix.at[fixed_index[:, None], index[None, :]].set(cross.conj().T)
        return matrix, scales

    def _dynamic_data_with_scales(self, fit_values, scales):
        _, dynamic = self._component_partitions()
        return jnp.stack(
            [
                jnp.asarray(
                    self.components[i].function(
                        self.data,
                        self._dynamic_parameter_mapping(
                            self.components[i].name, fit_values
                        ),
                    )
                )
                * scales[local]
                for local, i in enumerate(dynamic)
            ],
            axis=1,
        )

    @property
    def floating_dynamics(self) -> tuple[Parameter, ...]:
        """The DYNAMICS parameters that are not fixed."""
        return tuple(
            p
            for p in self.parameters
            if p.kind is ParameterKind.DYNAMICS and not p.fixed
        )

    @property
    def floating_dynamic_owners(self) -> frozenset[str]:
        """Names of components owning at least one floating DYNAMICS parameter."""
        return frozenset(p.owner for p in self.floating_dynamics if p.owner is not None)

    @property
    def is_compact(self) -> bool:
        """True if no component has a floating DYNAMICS parameter."""
        return not self.floating_dynamic_owners

    def check_parameters(self, parameters: Sequence[Parameter]) -> None:
        """Raise if ``parameters`` disagrees with the fixed/floating DYNAMICS
        split this cache was prepared with.

        ``prepare()`` bakes each DYNAMICS parameter's fixed-vs-floating status
        into the compact-vs-dynamic evaluation path at prepare time: a fixed
        component is folded into ``data_components``/``normalization_matrix_fixed``
        once and never re-reads ``fit_values`` again. If a later caller (e.g. a
        ``Minimizer`` built directly for advanced/low-level use, per
        ``docs/user_friendly_api.md``) supplies a *different* ``Parameter``
        sequence that marks the same name as floating, ``evaluate``/``amplitude``/
        ``normalization`` silently keep ignoring it — the gradient along that
        direction is a structural zero, not merely a small one, and Minuit will
        treat it as a flat direction with no error raised. Call this before
        handing a parameter list to ``Minimizer`` whenever it is not the exact
        object passed to ``prepare()``.
        """
        incoming = frozenset(
            p.owner
            for p in parameters
            if p.kind is ParameterKind.DYNAMICS and not p.fixed and p.owner is not None
        )
        if incoming == self.floating_dynamic_owners:
            return
        stale_fixed = sorted(incoming - self.floating_dynamic_owners)
        stale_floating = sorted(self.floating_dynamic_owners - incoming)
        raise ValueError(
            "parameters are inconsistent with the dynamics this cache was "
            "prepared with: components "
            f"{stale_fixed} are floating in `parameters` but were fixed when "
            f"this cache was prepared; components {stale_floating} are fixed "
            "in `parameters` but were floating at prepare time. Rebuild the "
            "cache with PreparedAmplitudeCache.prepare(..., parameters=parameters) "
            "or pass the same parameter list used to prepare the cache to Minimizer."
        )

    def coefficient_vector(self, fit_values: Mapping[str, object]) -> Array:
        """Resolve each component's complex coefficient at ``fit_values``."""
        return jnp.asarray(
            [
                coefficient_value(component.coefficient, fit_values)
                for component in self.components
            ]
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

    def _component_partitions(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if (
            self.fixed_component_indices is not None
            and self.dynamic_component_indices is not None
        ):
            return self.fixed_component_indices, self.dynamic_component_indices
        dynamic = tuple(
            index
            for index, component in enumerate(self.components)
            if component.name in self.floating_dynamic_owners
        )
        fixed = tuple(
            index for index in range(len(self.components)) if index not in dynamic
        )
        return fixed, dynamic

    def _evaluate_dynamic_components(
        self,
        fit_values: Mapping[str, object],
    ) -> tuple[Array | None, Array | None]:
        owners = self.floating_dynamic_owners
        if not owners:
            return None, None
        if self.normalization_chunks is not None:
            if self.data is None:
                raise RuntimeError("Dynamic cache is missing prepared event data")
            _, scales = self._chunked_dynamic_normalization(fit_values)
            data_values = self._dynamic_data_with_scales(fit_values, scales)
            _, dynamic = self._component_partitions()

            def evaluate_block(chunk):
                events, _, _, _ = chunk
                return jnp.stack(
                    [
                        jnp.asarray(
                            self.components[i].function(
                                events,
                                self._dynamic_parameter_mapping(
                                    self.components[i].name, fit_values
                                ),
                            )
                        )
                        for i in dynamic
                    ],
                    axis=1,
                )

            norm = jax.lax.map(evaluate_block, self.normalization_chunks)
            norm = norm.reshape((-1, len(dynamic)))[: self.normalization_weights.size]
            return data_values, norm * scales
        if self.data is None or self.normalization_data is None:
            raise RuntimeError("Dynamic cache is missing prepared event data")

        _, dynamic_indices = self._component_partitions()
        data_columns = []
        norm_columns = []
        for index in dynamic_indices:
            component = self.components[index]
            pars = self._dynamic_parameter_mapping(component.name, fit_values)
            data_values = jnp.asarray(component.function(self.data, pars))
            norm_values = jnp.asarray(component.function(self.normalization_data, pars))
            if _normalize_component(component, self.normalize_components):
                scale = self._single_component_scale(norm_values)
                data_values = data_values * scale
                norm_values = norm_values * scale
            data_columns.append(data_values)
            norm_columns.append(norm_values)

        return (
            jnp.stack(data_columns, axis=1),
            jnp.stack(norm_columns, axis=1),
        )

    def _amplitude_from_dynamic(
        self,
        coefficients: Array,
        dynamic_data: Array | None,
    ) -> Array:
        fixed_indices, dynamic_indices = self._component_partitions()
        size = self.data_components.shape[0]
        amplitude = jnp.zeros((size,), dtype=jnp.complex128)
        if fixed_indices:
            fixed_index = jnp.asarray(fixed_indices, dtype=jnp.int32)
            amplitude = amplitude + self.data_components @ coefficients[fixed_index]
        if dynamic_indices:
            if dynamic_data is None:
                raise RuntimeError("Dynamic data components are required")
            dynamic_index = jnp.asarray(dynamic_indices, dtype=jnp.int32)
            amplitude = amplitude + dynamic_data @ coefficients[dynamic_index]
        return amplitude

    def _matrix_from_dynamic(
        self,
        dynamic_norm: Array | None,
    ) -> Array:
        owners = self.floating_dynamic_owners
        if not owners:
            return self.normalization_matrix_fixed
        if dynamic_norm is None:
            raise RuntimeError("Dynamic normalization components are required")
        if self.normalization_components is None and self.normalization_chunks is None:
            raise RuntimeError(
                "Dynamic cache is missing fixed normalization components"
            )

        fixed_indices, dynamic_indices = self._component_partitions()
        dynamic_index = jnp.asarray(dynamic_indices, dtype=jnp.int32)
        weights = self._pdf_weights()
        n_points = dynamic_norm.shape[0]
        matrix = self.normalization_matrix_fixed

        if fixed_indices:
            fixed_index = jnp.asarray(fixed_indices, dtype=jnp.int32)
            if self.normalization_chunks is not None:
                fixed_columns = tuple(
                    self.normalization_chunks[3][column].reshape(-1)[
                        : self.normalization_weights.size
                    ]
                    for column in range(len(fixed_indices))
                )
            elif isinstance(self.normalization_components, tuple):
                fixed_columns = self.normalization_components
            else:
                fixed_columns = tuple(
                    self.normalization_components[:, column]
                    for column in range(len(fixed_indices))
                )
            cross_columns = []
            for column, component_index in zip(
                fixed_columns, fixed_indices, strict=True
            ):
                scaled_fixed = (
                    jnp.asarray(column) * self.component_scales[component_index]
                )
                cross_columns.append(
                    jnp.einsum(
                        "n,nd,n->d",
                        weights,
                        jnp.conj(dynamic_norm),
                        scaled_fixed,
                    )
                    / n_points
                )
            dynamic_fixed = jnp.stack(cross_columns, axis=1)
            matrix = matrix.at[dynamic_index[:, None], fixed_index[None, :]].set(
                dynamic_fixed
            )
            matrix = matrix.at[fixed_index[:, None], dynamic_index[None, :]].set(
                jnp.conj(dynamic_fixed).T
            )

        dynamic_dynamic = (
            jnp.einsum(
                "n,nd,ne->de",
                weights,
                jnp.conj(dynamic_norm),
                dynamic_norm,
            )
            / n_points
        )
        matrix = matrix.at[dynamic_index[:, None], dynamic_index[None, :]].set(
            dynamic_dynamic
        )
        diagonal = jnp.real(jnp.diag(dynamic_dynamic))
        matrix = matrix.at[dynamic_index, dynamic_index].set(diagonal)
        return matrix

    def _evaluate_components(
        self,
        fit_values: Mapping[str, object],
    ) -> tuple[Array, Array | None]:
        """Compatibility helper returning full component matrices.

        The hot likelihood path avoids constructing these full matrices and
        evaluates only the floating-dynamics block.
        """

        if not self.floating_dynamic_owners:
            return self.data_components, None
        if self.normalization_components is None and self.normalization_chunks is None:
            raise RuntimeError("Dynamic cache is missing normalization components")

        fixed_indices, dynamic_indices = self._component_partitions()
        dynamic_data, dynamic_norm = self._evaluate_dynamic_components(fit_values)
        fixed_lookup = {
            component_index: column
            for column, component_index in enumerate(fixed_indices)
        }
        dynamic_lookup = {
            component_index: column
            for column, component_index in enumerate(dynamic_indices)
        }
        data_columns = []
        norm_columns = []
        for index in range(len(self.components)):
            if index in fixed_lookup:
                column = fixed_lookup[index]
                data_columns.append(self.data_components[:, column])
                if self.normalization_chunks is not None:
                    fixed_norm = self.normalization_chunks[3][column].reshape(-1)
                    fixed_norm = fixed_norm[: self.normalization_weights.size]
                elif isinstance(self.normalization_components, tuple):
                    fixed_norm = self.normalization_components[column]
                else:
                    fixed_norm = self.normalization_components[:, column]
                norm_columns.append(
                    jnp.asarray(fixed_norm) * self.component_scales[index]
                )
            else:
                column = dynamic_lookup[index]
                data_columns.append(dynamic_data[:, column])
                norm_columns.append(dynamic_norm[:, column])
        return jnp.stack(data_columns, axis=1), jnp.stack(norm_columns, axis=1)

    def _matrix_with_dynamic_blocks(self, norm_components: Array | None) -> Array:
        """Compatibility helper for callers supplying a full component matrix."""

        owners = self.floating_dynamic_owners
        if not owners:
            return self.normalization_matrix_fixed
        if norm_components is None:
            raise RuntimeError("Dynamic normalization components are required")

        return normalization_matrix(
            norm_components,
            self.normalization_weights,
            self.efficiency_normalization,
        )

    def evaluate(self, fit_values: Mapping[str, object]) -> tuple[Array, Array]:
        """Return ``(intensity, normalization)`` at ``fit_values``."""
        coefficients = self.coefficient_vector(fit_values)
        if not self.floating_dynamic_owners:
            amplitude = self.data_components @ coefficients
            intensity = jnp.abs(amplitude) ** 2
            normalization = matrix_normalization(
                coefficients, self.normalization_matrix_fixed
            )
            return intensity, normalization

        if self.normalization_chunks is not None:
            matrix, scales = self._chunked_dynamic_normalization(fit_values)
            dynamic_data = self._dynamic_data_with_scales(fit_values, scales)
            amplitude = self._amplitude_from_dynamic(coefficients, dynamic_data)
            return jnp.abs(amplitude) ** 2, matrix_normalization(coefficients, matrix)

        dynamic_data, dynamic_norm = self._evaluate_dynamic_components(fit_values)
        amplitude = self._amplitude_from_dynamic(coefficients, dynamic_data)
        intensity = jnp.abs(amplitude) ** 2
        normalization = matrix_normalization(
            coefficients,
            self._matrix_from_dynamic(dynamic_norm),
        )
        return intensity, normalization

    def amplitude(self, fit_values: Mapping[str, object]) -> Array:
        """Coherent amplitude ``A(x) = sum_i c_i F_i(x)`` on the data sample."""
        coefficients = self.coefficient_vector(fit_values)
        if not self.floating_dynamic_owners:
            return self.data_components @ coefficients
        if self.normalization_chunks is not None:
            _, scales = self._chunked_dynamic_normalization(fit_values)
            dynamic_data = self._dynamic_data_with_scales(fit_values, scales)
        else:
            dynamic_data, _ = self._evaluate_dynamic_components(fit_values)
        return self._amplitude_from_dynamic(coefficients, dynamic_data)

    def intensity(self, fit_values: Mapping[str, object]) -> Array:
        """Unnormalized intensity ``|A(x)|^2`` on the data sample."""
        return jnp.abs(self.amplitude(fit_values)) ** 2

    def normalization(self, fit_values: Mapping[str, object]) -> Array:
        """Total normalization integral ``c^dagger M c`` at ``fit_values``."""
        coefficients = self.coefficient_vector(fit_values)
        if not self.floating_dynamic_owners:
            return matrix_normalization(coefficients, self.normalization_matrix_fixed)
        if self.normalization_chunks is not None:
            matrix, _ = self._chunked_dynamic_normalization(fit_values)
            return matrix_normalization(coefficients, matrix)
        _, dynamic_norm = self._evaluate_dynamic_components(fit_values)
        return matrix_normalization(
            coefficients,
            self._matrix_from_dynamic(dynamic_norm),
        )

    def normalization_matrix(self, fit_values: Mapping[str, object]) -> Array:
        """Hermitian normalization matrix ``M_ij = integral conj(F_i) F_j dPhi``."""
        if not self.floating_dynamic_owners:
            return self.normalization_matrix_fixed
        if self.normalization_chunks is not None:
            matrix, _ = self._chunked_dynamic_normalization(fit_values)
            return matrix
        _, dynamic_norm = self._evaluate_dynamic_components(fit_values)
        return self._matrix_from_dynamic(dynamic_norm)

    def fit_fractions(self, fit_values: Mapping[str, object]) -> Array:
        """Per-component fit fractions from the coefficients and norm matrix."""
        return matrix_fit_fractions(
            self.coefficient_vector(fit_values),
            self.normalization_matrix(fit_values),
        )

    def _fraction_jacobian_arrays(self):
        """Dynamic kernel inputs; no fitted-event arrays are needed."""
        return (
            self.normalization_data,
            self.normalization_weights,
            self.normalization_components,
            self.normalization_matrix_fixed,
            self.efficiency_normalization,
            self.component_scales,
            self.normalization_chunks,
        )

    def _build_fraction_jacobian_kernel(self, parameter_names):
        """Compile reusable sequential VJP rows without retaining sample arrays."""
        components = self.components
        parameters = self.parameters
        normalize = self.normalize_components
        dynamics_microbatch_size = self.dynamics_microbatch_size
        fixed, dynamic = self._component_partitions()
        names = tuple(parameter_names)

        @jax.jit
        def kernel(values, arrays):
            data, weights, columns, matrix, efficiency, scales, chunks = arrays
            cache = PreparedAmplitudeCache(
                components=components,
                parameters=parameters,
                data=data,
                normalization_data=data,
                normalization_weights=weights,
                data_components=jnp.empty((0, 0)),
                normalization_components=columns,
                normalization_matrix_fixed=matrix,
                efficiency_normalization=efficiency,
                component_scales=scales,
                normalize_components=normalize,
                fixed_component_indices=fixed,
                dynamic_component_indices=dynamic,
                normalization_chunks=chunks,
                dynamics_microbatch_size=dynamics_microbatch_size,
            )

            def fractions(vector):
                patched = dict(values)
                patched.update(zip(names, vector, strict=True))
                return cache.fit_fractions(patched)

            vector = jnp.asarray([values[name] for name in names], dtype=float)
            output, pullback = jax.vjp(fractions, vector)
            return jax.lax.map(
                lambda row: pullback(row)[0],
                jnp.eye(output.size, dtype=output.dtype),
            )

        return kernel

    def interference_fractions(self, fit_values: Mapping[str, object]) -> Array:
        """Pairwise interference fractions from the coefficients and norm matrix."""
        return matrix_interference_fractions(
            self.coefficient_vector(fit_values),
            self.normalization_matrix(fit_values),
        )
