"""Benchmark a realistic fit with floating K*(892) pole mass and fixed width.

The benchmark targets the prepared mass-only resonance path: event kinematics,
angular factors and event-side Blatt-Weisskopf denominators are cached, while
pole-mass-dependent quantities are reevaluated at every likelihood call.

Example
-------
python benchmarks/benchmark_floating_mass.py \
  --events 100000 \
  --normalization-resolution 1000 \
  --repeats 20
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from dalitzplotfitter import (
    BaBarFlatte,
    DecayChannel,
    DecayModel,
    FitSession,
    LASS,
    NonResonant,
    Parameter,
    RealImag,
    Resonance,
    enable_x64,
)


enable_x64()

_BENCHMARK_VERSION = 1


def _coefficient(name: str, x: float, y: float, *, fixed: bool = False):
    return RealImag(
        Parameter.coefficient(
            f"{name}.x", x, fixed=fixed, owner=name
        ),
        Parameter.coefficient(
            f"{name}.y", y, fixed=fixed, owner=name
        ),
    )


def make_model(normalization_resolution: int) -> DecayModel:
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    kstar_mass = Parameter.dynamics(
        "Kstar892.mass",
        0.8958,
        owner="Kstar892",
        backend_name="mass",
        bounds=(0.86, 0.93),
        step=1e-4,
    )
    return DecayModel(
        channel,
        [
            Resonance(
                "Kstar892",
                (0, 2),
                _coefficient("Kstar892", 1.00, 0.00, fixed=True),
                mass=kstar_mass,
                width=0.0474,
                spin=1,
                resonance_radius=4.0,
                parent_radius=4.0,
            ),
            Resonance(
                "KpiS",
                (0, 2),
                _coefficient("KpiS", 1.40, -0.60),
                lineshape=LASS(2.07, 3.32, 1.8),
                mass=1.425,
                width=0.270,
                spin=0,
                resonance_radius=4.0,
                parent_radius=4.0,
            ),
            Resonance(
                "rho770",
                (1, 2),
                _coefficient("rho770", 0.65, 0.10),
                mass=0.7753,
                width=0.1491,
                spin=1,
                resonance_radius=4.0,
                parent_radius=4.0,
            ),
            Resonance(
                "f0_980",
                (1, 2),
                _coefficient("f0_980", -0.20, 1.00),
                lineshape=BaBarFlatte(),
                mass=0.965,
                width=0.0,
                spin=0,
                resonance_radius=4.0,
                parent_radius=4.0,
            ),
            NonResonant(_coefficient("NR", -0.50, 0.10)),
        ],
        normalization_method="square-dalitz",
        normalization_resolution=normalization_resolution,
        normalization_pair=(0, 2),
    )


def _block(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return value
    if isinstance(value, dict):
        for item in value.values():
            _block(item)
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            _block(item)
        return value
    for name in (
        "data_components",
        "normalization_components",
        "normalization_matrix_fixed",
    ):
        if hasattr(value, name):
            _block(getattr(value, name))
    return value


def _timed(function):
    start = time.perf_counter()
    result = function()
    _block(result)
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--normalization-resolution", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=260902)
    args = parser.parse_args()

    model = make_model(args.normalization_resolution)
    normalization_sample, normalization_grid_seconds = _timed(
        lambda: model.normalization_sample
    )
    data, data_generation_seconds = _timed(
        lambda: model.generate_phase_space(args.events, seed=args.seed)
    )
    session = FitSession(model, data)
    cache, prepared_cache_seconds = _timed(lambda: session.signal_cache)

    prepared_keys = set(cache.data or {}) | set(cache.normalization_data or {})
    mass_only_angular_keys = sum(
        key.endswith("_angular_prepared") for key in prepared_keys
    )
    mass_only_res_barrier_keys = sum(
        key.endswith("_res_barrier_denominator") for key in prepared_keys
    )
    mass_only_parent_barrier_keys = sum(
        key.endswith("_parent_barrier_denominator") for key in prepared_keys
    )

    minimizer = session.minimizer()
    _, names, fcn, grad = minimizer._backend()
    point = {parameter.name: float(parameter.value) for parameter in session.parameters}
    vector = [point[name] for name in names]

    start = time.perf_counter()
    first_nll = fcn(*vector)
    first_gradient = grad(*vector)
    first_value_and_gradient_seconds = time.perf_counter() - start

    times = []
    last_nll = first_nll
    mass_index = names.index("Kstar892.mass")
    for index in range(args.repeats):
        shifted = list(vector)
        shifted[mass_index] += 1e-6 * (index + 1)
        start = time.perf_counter()
        last_nll = fcn(*shifted)
        grad(*shifted)
        times.append(time.perf_counter() - start)

    payload = {
        "benchmark_version": _BENCHMARK_VERSION,
        "device": str(jax.devices()[0]),
        "platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "events": args.events,
        "normalization_resolution": args.normalization_resolution,
        "normalization_points": normalization_sample.size,
        "components": len(model.components),
        "free_parameters": len(names),
        "floating_parameter": "Kstar892.mass",
        "fixed_width": 0.0474,
        "cache_is_compact": bool(cache.is_compact),
        "normalization_grid_seconds": normalization_grid_seconds,
        "data_generation_seconds": data_generation_seconds,
        "prepared_cache_seconds": prepared_cache_seconds,
        "mass_only_angular_keys": mass_only_angular_keys,
        "mass_only_res_barrier_keys": mass_only_res_barrier_keys,
        "mass_only_parent_barrier_keys": mass_only_parent_barrier_keys,
        "first_value_and_gradient_seconds": first_value_and_gradient_seconds,
        "steady_value_and_gradient_seconds_mean": float(jnp.mean(jnp.asarray(times))),
        "steady_value_and_gradient_seconds_min": float(jnp.min(jnp.asarray(times))),
        "steady_value_and_gradient_seconds_max": float(jnp.max(jnp.asarray(times))),
        "first_nll": float(first_nll),
        "last_nll": float(last_nll),
        "first_gradient_norm": float(jnp.linalg.norm(jnp.asarray(first_gradient))),
    }
    print("FLOATING_MASS_BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
