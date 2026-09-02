"""Benchmark the two public toy-generation methods on B+ -> K+ pi+ pi-.

The benchmark reports accept-reject end-to-end generation, inverse-transform
preparation plus first generation, and generation from an already prepared
inverse sampler. The latter is the relevant throughput for toy campaigns with
fixed truth parameters.
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
    prepare_inverse_toy_generator,
)


enable_x64()


class CountingModel:
    """Transparent proxy that counts weighted PhaseSpaceMC proposal work."""

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
    phase_space_calls: int | None
    phase_space_points: int | None
    output_per_proposal: float | None
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
    return DecayModel(
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


def block(sample) -> None:
    jax.block_until_ready(sample.s12)
    jax.block_until_ready(sample.s13)
    jax.block_until_ready(sample.s23)


def result_from_sample(
    method: str,
    sample,
    seconds: float,
    *,
    phase_space_calls: int | None = None,
    phase_space_points: int | None = None,
) -> Result:
    s12 = np.asarray(sample.s12)
    output_per_proposal = None
    if phase_space_points:
        output_per_proposal = sample.size / phase_space_points
    return Result(
        method=method,
        size=sample.size,
        seconds=seconds,
        events_per_second=sample.size / seconds,
        phase_space_calls=phase_space_calls,
        phase_space_points=phase_space_points,
        output_per_proposal=output_per_proposal,
        unique_s12_fraction=float(np.unique(s12).size / sample.size),
    )


def run_accept(model, size: int, seed: int):
    counted = CountingModel(model)
    start = time.perf_counter()
    sample = generate_toy(counted, size, seed=seed, method="accept-reject")
    block(sample)
    seconds = time.perf_counter() - start
    return result_from_sample(
        "accept-reject",
        sample,
        seconds,
        phase_space_calls=counted.phase_space_calls,
        phase_space_points=counted.phase_space_points,
    ), sample


def run_inverse(model, size: int, seed: int, resolution: int):
    start = time.perf_counter()
    prepared = prepare_inverse_toy_generator(model, resolution=resolution)
    preparation_seconds = time.perf_counter() - start

    start = time.perf_counter()
    sample = prepared.generate(size, seed=seed)
    block(sample)
    generation_seconds = time.perf_counter() - start
    total_seconds = preparation_seconds + generation_seconds
    total = result_from_sample("inverse-transform-total", sample, total_seconds)
    generated = result_from_sample("inverse-transform-prepared", sample, generation_seconds)
    return total, generated, sample, preparation_seconds


def projection_summary(first, second) -> dict[str, dict[str, float]]:
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
        result[variable] = {
            "mean_accept_reject": mean_a,
            "mean_inverse_transform": mean_b,
            "mean_difference_sigma": float(z_mean),
            "projection_total_variation": 0.5 * float(np.abs(pa - pb).sum()),
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
    parser.add_argument("--inverse-resolution", type=int, default=1024)
    args = parser.parse_args()

    model = make_model()

    # Warm up JAX work outside the timed region.
    warm, _ = run_accept(model, 2_000, args.seed - 1)
    del warm

    accept, accept_sample = run_accept(model, args.size, args.seed)
    inverse_total, inverse_prepared, inverse_sample, preparation_seconds = run_inverse(
        model, args.size, args.seed + 1, args.inverse_resolution
    )

    payload = {
        "device": str(jax.devices()[0]),
        "inverse_resolution": args.inverse_resolution,
        "inverse_preparation_seconds": preparation_seconds,
        "results": [
            asdict(accept),
            asdict(inverse_total),
            asdict(inverse_prepared),
        ],
        "closure": projection_summary(accept_sample, inverse_sample),
        "dalitz_2d_total_variation": dalitz_total_variation(
            accept_sample, inverse_sample
        ),
    }
    print("TOY_BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
