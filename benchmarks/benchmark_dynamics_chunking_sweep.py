"""Sweep ``normalization_chunk_size`` x ``dynamics_microbatch_size`` and report
their effect on fit-evaluation wall time and retained cache memory.

The model has two floating-mass/width resonances using ordinary parametric
lineshapes (``GounarisSakurai``/the default relativistic Breit-Wigner), so the
AD microbatching path (`PreparedAmplitudeCache._chunked_dynamic_normalization`,
see "AD microbatching for floating-dynamics normalization on constrained GPUs"
in docs/performance.md) is exercised without QMI's extra order-dependent
preparation constraint.

Each combination is padded internally the same way the real cache pads a
macro-chunk to a multiple of the microbatch size (see
``dalitzplotfitter.amplitude.cache._repeat_first_padded``): when
``dynamics_microbatch_size`` does not evenly divide ``normalization_chunk_size``,
part of every macro-chunk's reverse-AD pass is spent on discarded padding
rather than real points. ``padding_overhead_fraction`` in each result reports
that waste so it is not confused with genuine scan/checkpoint overhead.

Example
-------
python benchmarks/benchmark_dynamics_chunking_sweep.py --events 100000 \
    --normalization-resolution 500 \
    --chunk-sizes 50000,100000,200000 \
    --microbatch-sizes 20000,25000,40000,50000,100000
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time

import jax
import numpy as np

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    FitSession,
    GounarisSakurai,
    Parameter,
    RealImag,
    Resonance,
    enable_x64,
)


enable_x64()


def _coefficient(name: str, x: float, y: float, *, fixed: bool = False):
    return RealImag(
        Parameter.coefficient(f"{name}.x", x, owner=name, fixed=fixed),
        Parameter.coefficient(f"{name}.y", y, owner=name, fixed=fixed),
    )


def make_model(
    normalization_resolution: int,
    normalization_chunk_size: int,
    dynamics_microbatch_size: int,
) -> DecayModel:
    channel = DecayChannel("B+", ("pi+", "pi+", "pi-"))
    return DecayModel(
        channel,
        [
            Resonance(
                "rho770", (0, 2), _coefficient("rho770", 1.0, 0.0, fixed=True),
                mass=0.7708, width=0.1534, spin=1,
                lineshape=GounarisSakurai(),
                resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "omega782", (0, 2), _coefficient("omega782", 0.091, -0.007),
                mass=Parameter.dynamics(
                    "omega782.mass", 0.78265, owner="omega782", bounds=(0.70, 0.90)
                ),
                width=Parameter.dynamics(
                    "omega782.width", 0.00849, owner="omega782", bounds=(0.001, 0.05)
                ),
                spin=1,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "f2_1270", (0, 2), _coefficient("f2_1270", 0.291, 0.204),
                mass=1.2755, width=0.1867, spin=2,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "rho1450", (0, 2), _coefficient("rho1450", -0.223, 0.191),
                mass=Parameter.dynamics(
                    "rho1450.mass", 1.465, owner="rho1450", bounds=(1.30, 1.60)
                ),
                width=Parameter.dynamics(
                    "rho1450.width", 0.400, owner="rho1450", bounds=(0.20, 0.60)
                ),
                spin=1,
                resonance_radius=4.0, parent_radius=4.0,
            ),
        ],
        normalize_components=True,
        normalization_method="square-dalitz",
        normalization_resolution=normalization_resolution,
        normalization_pair=(0, 2),
        normalization_chunk_size=normalization_chunk_size,
        dynamics_microbatch_size=dynamics_microbatch_size,
    )


def _array_payload_bytes(value, seen=None):
    if seen is None:
        seen = set()
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(_array_payload_bytes(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_payload_bytes(v, seen) for v in value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        oid = id(value)
        if oid in seen:
            return 0
        seen.add(oid)
        return int(value.size * value.dtype.itemsize)
    return 0


def _block(value):
    for leaf in jax.tree_util.tree_leaves(value):
        blocker = getattr(leaf, "block_until_ready", None)
        if blocker is not None:
            blocker()
    return value


def _padding_overhead_fraction(
    point_count: int,
    chunk_size: int,
    microbatch_size: int,
) -> float:
    """Fraction of all microbatched work spent on inner and outer padding.

    The final macro-chunk is padded to the configured macro width, then every
    macro-chunk is padded to a multiple of its inner microbatch width.
    """

    macro = min(chunk_size, point_count)
    macro_count = math.ceil(point_count / macro)
    micro = min(microbatch_size, macro)
    microbatch_count = math.ceil(macro / micro)
    processed = macro_count * microbatch_count * micro
    return processed / point_count - 1.0


def _parse_int_list(text: str) -> list[int]:
    return [int(token) for token in text.split(",") if token.strip()]


def run_one(
    *,
    events: int,
    normalization_resolution: int,
    normalization_chunk_size: int,
    dynamics_microbatch_size: int,
    repeats: int,
    seed: int,
) -> dict:
    run_start = time.perf_counter()
    start = time.perf_counter()
    model = make_model(
        normalization_resolution, normalization_chunk_size, dynamics_microbatch_size
    )
    data = model.generate_phase_space(events, seed=seed, include_momenta=False)
    session = FitSession(model, data)
    cache = session.signal_cache
    _block(cache.data_components)
    _block(cache.normalization_components)
    _block(cache.normalization_chunks)
    cache_prepare_seconds = time.perf_counter() - start

    minimizer = session.minimizer()
    free, names, fcn, grad, _ = minimizer._backend()
    point = np.asarray([parameter.value for parameter in free], dtype=float)
    mass_index = names.index("omega782.mass")

    start = time.perf_counter()
    first_value = fcn(*point)
    first_gradient = grad(*point)
    first_seconds = time.perf_counter() - start

    times = []
    for i in range(repeats):
        shifted = point.copy()
        shifted[mass_index] += 1e-6 * (i + 1)
        start = time.perf_counter()
        fcn(*shifted)
        grad(*shifted)
        times.append(time.perf_counter() - start)

    retained = {
        "prepared_data": _array_payload_bytes(cache.data),
        "prepared_normalization_data": _array_payload_bytes(cache.normalization_data),
        "data_components": _array_payload_bytes(cache.data_components),
        "normalization_components": _array_payload_bytes(
            cache.normalization_components
        ),
        "normalization_chunks": _array_payload_bytes(cache.normalization_chunks),
        "normalization_weights": _array_payload_bytes(cache.normalization_weights),
        "normalization_matrix": _array_payload_bytes(cache.normalization_matrix_fixed),
    }
    # Everything this combination spent: model/data/cache preparation, the first
    # (compiling) objective+gradient call, and every steady-state repeat.
    total_seconds = time.perf_counter() - run_start

    return {
        "events": events,
        "normalization_resolution": normalization_resolution,
        "normalization_chunk_size": normalization_chunk_size,
        "dynamics_microbatch_size": dynamics_microbatch_size,
        "padding_overhead_fraction": _padding_overhead_fraction(
            int(cache.normalization_weights.size),
            normalization_chunk_size,
            dynamics_microbatch_size,
        ),
        "cache_prepare_seconds": cache_prepare_seconds,
        "first_jitted_objective_seconds": first_seconds,
        "steady_objective_seconds_mean": sum(times) / len(times),
        "steady_objective_seconds_min": min(times),
        "steady_objective_seconds_max": max(times),
        "total_seconds": total_seconds,
        "retained_cache_bytes_by_category": retained,
        "retained_cache_bytes_total": sum(retained.values()),
        "nll": float(first_value),
        "gradient_norm": float(np.linalg.norm(np.asarray(first_gradient, dtype=float))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--normalization-resolution", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=260905)
    parser.add_argument(
        "--chunk-sizes",
        type=str,
        default="50000,100000,200000",
        help="Comma-separated normalization_chunk_size values to sweep.",
    )
    parser.add_argument(
        "--microbatch-sizes",
        type=str,
        default="20000,25000,40000,50000,100000",
        help="Comma-separated dynamics_microbatch_size values to sweep.",
    )
    args = parser.parse_args()

    chunk_sizes = _parse_int_list(args.chunk_sizes)
    microbatch_sizes = _parse_int_list(args.microbatch_sizes)
    if args.events < 1 or args.normalization_resolution < 2 or args.repeats < 1:
        parser.error("events/repeats must be positive and resolution at least 2")
    if not chunk_sizes or min(chunk_sizes) < 1:
        parser.error("--chunk-sizes must contain positive integers")
    if not microbatch_sizes or min(microbatch_sizes) < 1:
        parser.error("--microbatch-sizes must contain positive integers")

    results = []
    for chunk_size in chunk_sizes:
        for microbatch_size in microbatch_sizes:
            result = run_one(
                events=args.events,
                normalization_resolution=args.normalization_resolution,
                normalization_chunk_size=chunk_size,
                dynamics_microbatch_size=microbatch_size,
                repeats=args.repeats,
                seed=args.seed,
            )
            print(f"SWEEP_RESULT_JSON={json.dumps(result)}")
            results.append(result)
            # Each point should pay its own cold-compilation cost and release
            # executables before the next configuration is measured.
            gc.collect()
            jax.clear_caches()

    header = (
        f"{'chunk':>10}  {'microbatch':>10}  {'pad_%':>7}  "
        f"{'prepare_s':>10}  {'first_s':>9}  {'steady_s_mean':>14}  "
        f"{'total_s':>9}  {'retained_MB':>12}"
    )
    print()
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['normalization_chunk_size']:>10}  "
            f"{result['dynamics_microbatch_size']:>10}  "
            f"{100 * result['padding_overhead_fraction']:>6.1f}%  "
            f"{result['cache_prepare_seconds']:>10.3f}  "
            f"{result['first_jitted_objective_seconds']:>9.3f}  "
            f"{result['steady_objective_seconds_mean']:>14.6f}  "
            f"{result['total_seconds']:>9.3f}  "
            f"{result['retained_cache_bytes_total'] / 1e6:>12.2f}"
        )


if __name__ == "__main__":
    main()
