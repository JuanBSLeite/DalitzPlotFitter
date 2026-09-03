"""Decompose cold amplitude-cache preparation into compile and execution stages.

This benchmark measures the fixed-size normalization chunk actually used by the
coefficient-only cache, then the complete chunked normalization pass and the
separate data-side XLA program.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from dalitzplotfitter import (
    BaBarFlatte,
    DecayChannel,
    DecayModel,
    LASS,
    NonResonant,
    Parameter,
    RealImag,
    Resonance,
    enable_x64,
)
from dalitzplotfitter.amplitude.cache import (
    DEFAULT_NORMALIZATION_CHUNK_SIZE,
    _compact_data_kernel,
    _compact_normalization_kernel,
    _padded_mapping_chunk,
    _padded_vector_chunk,
)


enable_x64()

_BENCHMARK_VERSION = 2


def _coefficient(name: str, x: float, y: float, *, fixed: bool = False):
    return RealImag(
        Parameter.coefficient(f"{name}.x", x, fixed=fixed, owner=name),
        Parameter.coefficient(f"{name}.y", y, fixed=fixed, owner=name),
    )


def make_model(normalization_resolution: int) -> DecayModel:
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    return DecayModel(
        channel,
        [
            Resonance(
                "Kstar892", (0, 2), _coefficient("Kstar892", 1.00, 0.00, fixed=True),
                mass=0.8958, width=0.0474, spin=1, resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "KpiS", (0, 2), _coefficient("KpiS", 1.40, -0.60),
                lineshape=LASS(2.07, 3.32, 1.8), mass=1.425, width=0.270, spin=0,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "rho770", (1, 2), _coefficient("rho770", 0.65, 0.10),
                mass=0.7753, width=0.1491, spin=1, resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "f0_980", (1, 2), _coefficient("f0_980", -0.20, 1.00),
                lineshape=BaBarFlatte(), mass=0.965, width=0.0, spin=0,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            NonResonant(_coefficient("NR", -0.50, 0.10)),
        ],
        normalization_method="square-dalitz",
        normalization_resolution=normalization_resolution,
        normalization_pair=(0, 2),
    )


def _block(value):
    for leaf in jax.tree_util.tree_leaves(value):
        blocker = getattr(leaf, "block_until_ready", None)
        if blocker is not None:
            blocker()
    return value


def _timed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def _timed_blocking(function):
    start = time.perf_counter()
    result = function()
    _block(result)
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--normalization-resolution", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260902)
    parser.add_argument(
        "--normalization-chunk-size",
        type=int,
        default=DEFAULT_NORMALIZATION_CHUNK_SIZE,
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.normalization_chunk_size < 1:
        raise ValueError("repeats and normalization chunk size must be positive")

    model = make_model(args.normalization_resolution)
    normalization_sample, normalization_grid_seconds = _timed_blocking(
        lambda: model.normalization_sample
    )
    data, data_generation_seconds = _timed_blocking(
        lambda: model.generate_phase_space(args.events, seed=args.seed)
    )

    components = model.amplitude_model.components
    norm_data = normalization_sample.as_dict()
    weights = jnp.asarray(normalization_sample.weights)
    efficiency = jnp.ones_like(weights)
    data_dict = data.as_dict()

    normalization_kernel = _compact_normalization_kernel(
        components,
        normalize_components=model.normalize_components,
        has_efficiency=False,
        chunk_size=args.normalization_chunk_size,
    )
    chunk_kernel = normalization_kernel.chunk_kernel
    chunk_size = min(args.normalization_chunk_size, normalization_sample.size)
    chunk_data = _padded_mapping_chunk(norm_data, 0, chunk_size, chunk_size)
    chunk_weights = _padded_vector_chunk(
        weights, 0, chunk_size, chunk_size, padding_value=0.0
    )
    chunk_efficiency = _padded_vector_chunk(
        efficiency, 0, chunk_size, chunk_size, padding_value=1.0
    )

    lowered_norm, norm_lower_seconds = _timed(
        lambda: chunk_kernel.lower(chunk_data, chunk_weights, chunk_efficiency)
    )
    compiled_norm, norm_compile_seconds = _timed(lowered_norm.compile)
    _, norm_chunk_first_execute_seconds = _timed_blocking(
        lambda: compiled_norm(chunk_data, chunk_weights, chunk_efficiency)
    )

    chunk_warm_times = []
    for _ in range(args.repeats):
        _, elapsed = _timed_blocking(
            lambda: compiled_norm(chunk_data, chunk_weights, chunk_efficiency)
        )
        chunk_warm_times.append(elapsed)

    norm_outputs, norm_total_first_seconds = _timed_blocking(
        lambda: normalization_kernel(norm_data, weights, efficiency)
    )
    fixed_matrix, diagonal, scales = norm_outputs
    norm_total_warm_times = []
    for _ in range(args.repeats):
        _, elapsed = _timed_blocking(
            lambda: normalization_kernel(norm_data, weights, efficiency)
        )
        norm_total_warm_times.append(elapsed)

    data_kernel = _compact_data_kernel(
        components,
        normalize_components=model.normalize_components,
    )
    lowered_data, data_lower_seconds = _timed(
        lambda: data_kernel.lower(data_dict, scales)
    )
    compiled_data, data_compile_seconds = _timed(lowered_data.compile)
    data_components, data_first_execute_seconds = _timed_blocking(
        lambda: compiled_data(data_dict, scales)
    )
    data_warm_times = []
    for _ in range(args.repeats):
        _, elapsed = _timed_blocking(lambda: compiled_data(data_dict, scales))
        data_warm_times.append(elapsed)

    diagonal_host = np.asarray(jax.device_get(diagonal), dtype=float)
    payload = {
        "benchmark_version": _BENCHMARK_VERSION,
        "device": str(jax.devices()[0]),
        "platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "events": args.events,
        "normalization_resolution": args.normalization_resolution,
        "normalization_points": normalization_sample.size,
        "normalization_chunk_size": chunk_size,
        "normalization_chunks": int(np.ceil(normalization_sample.size / chunk_size)),
        "components": len(components),
        "normalization_grid_seconds": normalization_grid_seconds,
        "data_generation_seconds": data_generation_seconds,
        "normalization_chunk_lower_seconds": norm_lower_seconds,
        "normalization_chunk_compile_seconds": norm_compile_seconds,
        "normalization_chunk_first_execute_seconds": norm_chunk_first_execute_seconds,
        "normalization_chunk_warm_execute_seconds_mean": float(np.mean(chunk_warm_times)),
        "normalization_total_first_seconds": norm_total_first_seconds,
        "normalization_total_warm_seconds_mean": float(np.mean(norm_total_warm_times)),
        "normalization_total_warm_seconds_min": float(np.min(norm_total_warm_times)),
        "normalization_total_warm_seconds_max": float(np.max(norm_total_warm_times)),
        "data_lower_seconds": data_lower_seconds,
        "data_compile_seconds": data_compile_seconds,
        "data_first_execute_seconds": data_first_execute_seconds,
        "data_warm_execute_seconds_mean": float(np.mean(data_warm_times)),
        "normalization_matrix_diagonal_min": float(np.min(diagonal_host)),
        "normalization_matrix_diagonal_max": float(np.max(diagonal_host)),
        "normalization_matrix_finite": bool(
            np.all(np.isfinite(np.asarray(jax.device_get(fixed_matrix))))
        ),
        "data_components_finite": bool(
            np.all(np.isfinite(np.asarray(jax.device_get(data_components))))
        ),
    }
    print("CACHE_STAGE_BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
