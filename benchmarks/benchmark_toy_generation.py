"""Benchmark the high-level toy generators on the full B+ -> K+ pi+ pi- model.

The benchmark intentionally measures end-to-end generation after model setup.
For accept-reject, the proposal count includes the pilot sample because it is a
real part of the generation cost. For the legacy resampler the pool factor is
kept at 10, matching its historical default scaling.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import jax
import numpy as np

from dalitzplotfitter import (
    BaBarFlatte,
    DecayChannel,
    DecayModel,
    LASS,
    NonResonant,
    RealImag,
    Resonance,
    enable_x64,
    generate_toy,
)


enable_x64()


class CountingModel:
    """Transparent model proxy that counts phase-space proposal work."""

    def __init__(self, model):
        self._model = model
        self.phase_space_calls = 0
        self.phase_space_points = 0

    def __getattr__(self, name):
        return getattr(self._model, name)

    def generate_phase_space(self, size, *, seed=None):
        self.phase_space_calls += 1
        self.phase_space_points += int(size)
        return self._model.generate_phase_space(size, seed=seed)


@dataclass
class Result:
    method: str
    size: int
    seconds: float
    events_per_second: float
    phase_space_calls: int
    phase_space_points: int
    output_per_proposal: float
    unique_s12_fraction: float


def make_model():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    coefficients = {
        "Kstar892": RealImag(1.00, 0.00),
        "KpiS": RealImag(1.40, -0.60),
        "rho770": RealImag(0.65, 0.10),
        "f0_980": RealImag(-0.20, 1.00),
        "NR": RealImag(-0.50, 0.10),
    }
    model = DecayModel(
        channel,
        [
            Resonance(
                "Kstar892", (0, 2), coefficients["Kstar892"],
                mass=0.8958, width=0.0474, spin=1,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "KpiS", (0, 2), coefficients["KpiS"],
                lineshape=LASS(2.07, 3.32, 1.8),
                mass=1.425, width=0.270, spin=0,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "rho770", (1, 2), coefficients["rho770"],
                mass=0.7753, width=0.1491, spin=1,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            Resonance(
                "f0_980", (1, 2), coefficients["f0_980"],
                lineshape=BaBarFlatte(), mass=0.965, width=0.0, spin=0,
                resonance_radius=4.0, parent_radius=4.0,
            ),
            NonResonant(coefficients["NR"]),
        ],
        normalization_method="square-dalitz",
        normalization_resolution=80,
        normalization_pair=(0, 2),
    )
    return model


def block(sample) -> None:
    """Ensure asynchronous JAX work is complete before stopping the timer."""

    jax.block_until_ready(sample.s12)
    jax.block_until_ready(sample.s13)
    jax.block_until_ready(sample.s23)


def run_one(model, size: int, method: str, seed: int, pool_factor: int) -> tuple[Result, object]:
    counted = CountingModel(model)
    kwargs = {}
    if method == "resample":
        kwargs["pool_size"] = pool_factor * size

    start = time.perf_counter()
    sample = generate_toy(
        counted,
        size,
        seed=seed,
        method=method,
        **kwargs,
    )
    block(sample)
    seconds = time.perf_counter() - start

    s12 = np.asarray(sample.s12)
    unique_fraction = float(np.unique(s12).size / size)
    result = Result(
        method=method,
        size=size,
        seconds=seconds,
        events_per_second=size / seconds,
        phase_space_calls=counted.phase_space_calls,
        phase_space_points=counted.phase_space_points,
        output_per_proposal=size / counted.phase_space_points,
        unique_s12_fraction=unique_fraction,
    )
    return result, sample


def projection_summary(first, second) -> dict[str, dict[str, float]]:
    """Statistical closure checks between two unweighted generated samples."""

    result = {}
    for variable in ("s12", "s13", "s23"):
        a = np.asarray(getattr(first, variable), dtype=float)
        b = np.asarray(getattr(second, variable), dtype=float)
        mean_a = float(a.mean())
        mean_b = float(b.mean())
        variance = a.var(ddof=1) / a.size + b.var(ddof=1) / b.size
        z_mean = (mean_a - mean_b) / np.sqrt(variance) if variance > 0.0 else 0.0

        lo = min(float(a.min()), float(b.min()))
        hi = max(float(a.max()), float(b.max()))
        ha, edges = np.histogram(a, bins=80, range=(lo, hi))
        hb, _ = np.histogram(b, bins=edges)
        pa = ha / ha.sum()
        pb = hb / hb.sum()
        total_variation = 0.5 * float(np.abs(pa - pb).sum())
        result[variable] = {
            "mean_accept_reject": mean_a,
            "mean_resample": mean_b,
            "mean_difference_sigma": float(z_mean),
            "projection_total_variation": total_variation,
        }
    return result


def dalitz_total_variation(first, second) -> float:
    a13 = np.asarray(first.s13)
    a23 = np.asarray(first.s23)
    b13 = np.asarray(second.s13)
    b23 = np.asarray(second.s23)
    x_range = (min(a13.min(), b13.min()), max(a13.max(), b13.max()))
    y_range = (min(a23.min(), b23.min()), max(a23.max(), b23.max()))
    ha, x_edges, y_edges = np.histogram2d(a13, a23, bins=40, range=(x_range, y_range))
    hb, _, _ = np.histogram2d(b13, b23, bins=(x_edges, y_edges))
    pa = ha / ha.sum()
    pb = hb / hb.sum()
    return 0.5 * float(np.abs(pa - pb).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=260902)
    parser.add_argument("--pool-factor", type=int, default=10)
    args = parser.parse_args()

    model = make_model()

    # Warm up both numerical paths outside the timed region.
    warm_accept, _ = run_one(model, 2_000, "accept-reject", args.seed - 2, 10)
    warm_resample, _ = run_one(model, 2_000, "resample", args.seed - 1, 10)
    del warm_accept, warm_resample

    accept, accept_sample = run_one(
        model, args.size, "accept-reject", args.seed, args.pool_factor
    )
    resample, resample_sample = run_one(
        model, args.size, "resample", args.seed + 1, args.pool_factor
    )

    payload = {
        "device": str(jax.devices()[0]),
        "pool_factor": args.pool_factor,
        "results": [asdict(accept), asdict(resample)],
        "closure": projection_summary(accept_sample, resample_sample),
        "dalitz_2d_total_variation": dalitz_total_variation(
            accept_sample, resample_sample
        ),
    }
    print("TOY_BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
