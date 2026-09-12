"""Benchmark fit-fraction uncertainty propagation, including cache preparation.

Run each method in a fresh process for comparable first-call timings:
  python benchmarks/benchmark_fit_fraction_errors.py --method eager --require-gpu
  python benchmarks/benchmark_fit_fraction_errors.py --method prepared --require-gpu

Uses model starting values and a synthetic covariance, not a fitted data result.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import jax
import numpy as np
from benchmark_fit_evaluation import make_model as make_isobar
from benchmark_qmi_memory_speed import make_model as make_qmi

from dalitzplotfitter import delta_method_errors, enable_x64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("prepared", "eager"), default="prepared")
    parser.add_argument("--model", choices=("qmi", "isobar"), default="qmi")
    parser.add_argument("--normalization-resolution", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    enable_x64()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU required; refusing CPU fallback")
    factory = make_qmi if args.model == "qmi" else make_isobar
    model = factory(args.normalization_resolution)
    sample = model.normalization_sample
    sample.weights.block_until_ready()
    values = {p.name: p.value for p in model.parameters}
    names = tuple(p.name for p in model.parameters if not p.fixed)
    covariance = np.eye(len(names))*1e-4
    runs = []
    for repeat in range(args.repeats):
        point = dict(values)
        point[names[0]] += repeat*1e-5
        start = perf_counter()
        if args.method == "prepared":
            errors = model.fit_fraction_errors(point, covariance, names)
        else:
            # Reference algorithm: full data-side preparation plus eager VJP
            # rows, as used before the dedicated fit-fraction Jacobian kernel.
            cache = model.prepare_cache(sample, normalization_sample=sample)
            array = np.asarray(delta_method_errors(
                cache.fit_fractions, point, names, covariance,
            ))
            errors = dict(zip((c.name for c in cache.components), array, strict=True))
            del cache
        elapsed = perf_counter() - start
        runs.append({"seconds": elapsed, "errors": errors})
    print(json.dumps({
        "method": args.method,
        "model": args.model,
        "device": str(jax.devices()[0]),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "free_parameters": len(names),
        "normalization_points": sample.size,
        "runs": runs,
        "memory_stats": jax.devices()[0].memory_stats(),
    }, indent=2))


if __name__ == "__main__":
    main()
