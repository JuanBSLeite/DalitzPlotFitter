"""Compare numerical and JAX HESSE at a fixed Dalitz-model parameter point.

This measures curvature evaluation, not fit convergence. Phase-space data and
initial model parameters need not define a minimum; covariance status is reported
explicitly. Cold and warm timings are separate, and every repetition changes the
point to avoid timing the host Hessian cache.

python benchmarks/benchmark_hesse.py --model qmi --events 5000 --repeats 3
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import jax
import numpy as np
from benchmark_fit_evaluation import make_model as make_isobar_model
from benchmark_qmi_memory_speed import make_model as make_qmi_model
from iminuit import Minuit

from dalitzplotfitter import FitSession, enable_x64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("isobar", "qmi"), default="qmi")
    parser.add_argument("--events", type=int, default=5000)
    parser.add_argument("--normalization-resolution", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    enable_x64()
    factory = make_qmi_model if args.model == "qmi" else make_isobar_model
    model = factory(args.normalization_resolution)
    session = FitSession(model, model.generate_phase_space(args.events, seed=131))
    minimizer = session.minimizer()
    free, names, fcn, grad, hessian = minimizer._backend()
    point = np.array([p.value for p in free])
    started = perf_counter()
    fcn(*point)
    value_gradient_cold = perf_counter() - started
    payload = {
        "model": args.model,
        "events": args.events,
        "normalization_resolution": args.normalization_resolution,
        "free_parameters": len(free),
        "device": str(jax.devices()[0]),
        "value_gradient_cold_seconds": value_gradient_cold,
        "modes": {},
    }
    for mode in ("numerical", "jax"):
        runs = []
        for repeat in range(args.repeats + 1):
            shifted = point.copy()
            shifted[0] += 1e-5 * (repeat + 1)
            result = Minuit(
                fcn, *shifted, name=names, grad=grad,
                hessian=hessian if mode == "jax" else None,
                g2=(lambda *v: np.diag(hessian(*v))) if mode == "jax" else None,
            )
            result.errordef = 0.5
            result.strategy = 2
            for parameter in free:
                if parameter.bounds is not None:
                    result.limits[parameter.name] = parameter.bounds
                if parameter.step is not None:
                    result.errors[parameter.name] = parameter.step
            started = perf_counter()
            result.hesse()
            runs.append({
                "seconds": perf_counter() - started,
                "nfcn": result.nfcn,
                "ngrad": result.ngrad,
                "nhessian": result.nhessian,
                "accurate_covariance": bool(result.fmin.has_accurate_covar),
                "forced_positive_definite": bool(result.fmin.has_made_posdef_covar),
            })
        payload["modes"][mode] = {
            "cold": runs[0],
            "warm_median_seconds": float(np.median([r["seconds"] for r in runs[1:]])),
            "warm_runs": runs[1:],
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
