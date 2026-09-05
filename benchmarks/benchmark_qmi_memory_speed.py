"""Benchmark QMI cache memory and repeated objective evaluation.

This benchmark is intentionally focused on the dynamic-cache path used by the
B -> pi pi pi QMI fits: one floating scalar QMI component plus several fixed-
dynamics higher waves whose complex coefficients may float.

Example
-------
python benchmarks/benchmark_qmi_memory_speed.py --events 250000 --normalization-resolution 500
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    FitSession,
    GounarisSakurai,
    Parameter,
    QMI,
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


def make_model(normalization_resolution: int) -> DecayModel:
    channel = DecayChannel("B+", ("pi+", "pi+", "pi-"))
    threshold = channel.daughter_masses[0] + channel.daughter_masses[2]
    maximum = channel.parent_mass - channel.daughter_masses[1]
    knots = (
        float(threshold),
        0.40, 0.51, 0.63, 0.70, 0.77, 0.84, 0.90, 0.99, 1.11,
        1.21, 1.30, 1.40, 1.56, 1.74, 2.00, 2.50, 3.00, 3.50,
        float(maximum),
    )
    magnitudes = tuple(
        Parameter.dynamics(
            f"S_QMI.mag[{i}]",
            1.0,
            owner="S_QMI",
            bounds=(0.0, 10.0),
        )
        for i in range(len(knots))
    )
    phases = tuple(
        Parameter.dynamics(
            f"S_QMI.phase[{i}]",
            0.0,
            owner="S_QMI",
            bounds=(-8.0 * jnp.pi, 8.0 * jnp.pi),
        )
        for i in range(len(knots))
    )
    qmi = QMI(knots, magnitudes, phases, interpolation="linear")

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
                mass=0.78265, width=0.00849, spin=1,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "f2_1270", (0, 2), _coefficient("f2_1270", 0.291, 0.204),
                mass=1.2755, width=0.1867, spin=2,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "rho1450", (0, 2), _coefficient("rho1450", -0.223, 0.191),
                mass=1.465, width=0.400, spin=1,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "rho3_1690", (0, 2), _coefficient("rho3_1690", 0.073, -0.045),
                mass=1.6888, width=0.161, spin=3,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "S_QMI", (0, 2), RealImag(1.0, 0.0),
                mass=1.0, width=0.1, spin=0,
                lineshape=qmi,
                normalize_component=False,
                resonance_radius=4.0, parent_radius=4.0,
            ),
        ],
        normalize_components=True,
        normalization_method="square-dalitz",
        normalization_resolution=normalization_resolution,
        normalization_pair=(0, 2),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=250_000)
    parser.add_argument("--normalization-resolution", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=260905)
    args = parser.parse_args()

    model = make_model(args.normalization_resolution)
    data = model.generate_phase_space(
        args.events,
        seed=args.seed,
        include_momenta=False,
    )
    session = FitSession(model, data)

    start = time.perf_counter()
    cache = session.signal_cache
    _block(cache.data_components)
    _block(cache.normalization_components)
    cache_prepare_seconds = time.perf_counter() - start

    # Use the exact compiled value+gradient backend used by Minimizer rather
    # than wrapping the mapping-based objective in another jax.jit. The latter
    # can make the frozen FitSession/objective participate in JAX static
    # hashing, which is both unrepresentative of a real fit and fails when the
    # session contains mappings.
    minimizer = session.minimizer()
    free, names, fcn, grad = minimizer._backend()
    point = np.asarray([parameter.value for parameter in free], dtype=float)
    qmi_phase_index = names.index("S_QMI.phase[8]")

    fixed = {
        parameter.name: parameter.value
        for parameter in session.parameters
        if parameter.fixed
    }
    objective_mapping = session.objective

    def vector_mapping(vector):
        mapping = dict(fixed)
        mapping.update({name: vector[i] for i, name in enumerate(names)})
        return mapping

    def vector_objective(vector):
        return objective_mapping(vector_mapping(vector))

    def vector_dynamic(vector):
        return cache._evaluate_dynamic_components(vector_mapping(vector))

    value_only = jax.jit(vector_objective)
    dynamic_only = jax.jit(vector_dynamic)
    matrix_only = jax.jit(lambda dynamic_norm: cache._matrix_from_dynamic(dynamic_norm))

    point_device = jnp.asarray(point)

    start = time.perf_counter()
    first_value_only = value_only(point_device)
    first_value_only.block_until_ready()
    first_value_only_seconds = time.perf_counter() - start

    dynamic_data, dynamic_norm = dynamic_only(point_device)
    _block((dynamic_data, dynamic_norm))
    matrix_only(dynamic_norm).block_until_ready()

    forward_times = []
    matrix_times = []
    value_times = []
    for i in range(args.repeats):
        shifted = point.copy()
        shifted[qmi_phase_index] += 1e-6 * (i + 1)
        shifted_device = jnp.asarray(shifted)

        start = time.perf_counter()
        dyn_data, dyn_norm = dynamic_only(shifted_device)
        _block((dyn_data, dyn_norm))
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        matrix_only(dyn_norm).block_until_ready()
        matrix_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        value_only(shifted_device).block_until_ready()
        value_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    first = fcn(*point)
    first_gradient = grad(*point)
    first_seconds = time.perf_counter() - start

    times = []
    values = []
    for i in range(args.repeats):
        shifted = point.copy()
        shifted[qmi_phase_index] += 1e-6 * (i + 1)
        start = time.perf_counter()
        values.append(fcn(*shifted))
        grad(*shifted)
        times.append(time.perf_counter() - start)

    retained = {
        "prepared_data": _array_payload_bytes(cache.data),
        "prepared_normalization_data": _array_payload_bytes(cache.normalization_data),
        "data_components": _array_payload_bytes(cache.data_components),
        "normalization_components": _array_payload_bytes(cache.normalization_components),
        "normalization_weights": _array_payload_bytes(cache.normalization_weights),
        "normalization_matrix": _array_payload_bytes(cache.normalization_matrix_fixed),
    }
    payload = {
        "device": str(jax.devices()[0]),
        "events": args.events,
        "normalization_points": model.normalization_sample.size,
        "normalization_resolution": args.normalization_resolution,
        "qmi_knots": 20,
        "cache_prepare_seconds": cache_prepare_seconds,
        "first_jitted_objective_seconds": first_seconds,
        "first_value_only_seconds": first_value_only_seconds,
        "steady_qmi_forward_seconds_mean": sum(forward_times) / len(forward_times),
        "steady_qmi_forward_seconds_min": min(forward_times),
        "steady_qmi_forward_seconds_max": max(forward_times),
        "steady_matrix_update_seconds_mean": sum(matrix_times) / len(matrix_times),
        "steady_matrix_update_seconds_min": min(matrix_times),
        "steady_matrix_update_seconds_max": max(matrix_times),
        "steady_value_only_seconds_mean": sum(value_times) / len(value_times),
        "steady_value_only_seconds_min": min(value_times),
        "steady_value_only_seconds_max": max(value_times),
        "steady_objective_seconds_mean": sum(times) / len(times),
        "steady_objective_seconds_min": min(times),
        "steady_objective_seconds_max": max(times),
        "retained_cache_bytes_by_category": retained,
        "retained_cache_bytes_total": sum(retained.values()),
        "nll": float(first),
        "gradient_norm": float(np.linalg.norm(first_gradient)),
        "last_nll": float(values[-1]) if values else float(first),
    }
    print("QMI_MEMORY_SPEED_BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
