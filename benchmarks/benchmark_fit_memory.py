"""Measure live/peak GPU memory of a QMI fit in a fresh process.

Run before and after a change with identical arguments. Allocator counters do
not include all CUDA/runtime memory. The data are phase-space events, not a
physics closure sample; values/gradients/Hessians are numerical cross-checks.

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python benchmarks/benchmark_fit_memory.py --floating-mass --hessian
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from time import perf_counter

import jax
import numpy as np
from benchmark_qmi_memory_speed import make_model

from dalitzplotfitter import DecayModel, FitSession, Parameter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--normalization-resolution", type=int, default=1000)
    parser.add_argument("--floating-mass", action="store_true")
    parser.add_argument("--hessian", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.events < 1 or args.normalization_resolution < 2 or args.repeats < 1:
        parser.error("events/repeats must be positive and resolution at least 2")
    device = jax.devices()[0]
    stages = []
    start = perf_counter()

    def report(name):
        gc.collect()
        for array in jax.live_arrays():
            array.block_until_ready()
        stats = device.memory_stats() or {}
        stage = dict(
            stage=name,
            seconds=perf_counter() - start,
            bytes_in_use=stats.get("bytes_in_use"),
            peak_bytes_in_use=stats.get("peak_bytes_in_use"),
            pool_bytes=stats.get("pool_bytes"),
        )
        stages.append(stage)
        print(json.dumps(stage), flush=True)

    model = make_model(args.normalization_resolution, "none")
    if args.floating_mass:
        components = tuple(
            replace(
                c,
                mass=Parameter.dynamics(
                    "rho1450.mass",
                    1.465,
                    owner="rho1450",
                    bounds=(1.3, 1.6),
                ),
            )
            if c.name == "rho1450"
            else c
            for c in model.components
        )
        model = DecayModel(
            model.channel,
            components,
            normalize_components=True,
            normalization_method="square-dalitz",
            normalization_pair=(0, 2),
            normalization_resolution=args.normalization_resolution,
        )
    data = model.generate_phase_space(args.events, seed=260905, include_momenta=False)
    report("data")
    session = FitSession(model, data)
    cache = session.signal_cache
    report("cache")
    minimizer = session.minimizer(hessian="jax" if args.hessian else "numerical")
    free, names, fcn, grad, hessian = minimizer._backend()
    point = np.asarray([p.value for p in free])
    value = fcn(*point)
    gradient = np.asarray(grad(*point))
    report("value_and_gradient")
    times = []
    for i in range(args.repeats):
        shifted = point.copy()
        shifted[names.index("S_QMI.phase[8]")] += 1e-6 * (i + 1)
        tick = perf_counter()
        fcn(*shifted)
        grad(*shifted)
        times.append(perf_counter() - tick)
    curvature = np.asarray(hessian(*point)) if args.hessian else None
    report("hessian" if args.hessian else "warm_gradient")
    leaves = jax.tree_util.tree_leaves((cache.data, cache.normalization_chunks))
    payload = dict(
        device=device.device_kind,
        backend=device.platform,
        x64=jax.config.x64_enabled,
        events=data.size,
        normalization_points=model.normalization_sample.size,
        floating_mass=args.floating_mass,
        stages=stages,
        prepared_payload_bytes=sum(a.nbytes for a in leaves),
        warm_gradient_seconds=float(np.mean(times)),
        nll=float(value),
        parameter_names=names,
        gradient=gradient.tolist(),
        hessian=None if curvature is None else curvature.tolist(),
    )
    print("FIT_MEMORY_JSON=" + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
