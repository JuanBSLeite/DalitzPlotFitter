"""Benchmark retained memory for full and compact phase-space samples.

This benchmark reports the array payload size exposed by PhaseSpaceSample.nbytes.
It does not attempt to measure allocator fragmentation or XLA runtime buffers.

Example
-------
python benchmarks/benchmark_memory.py --events 1000000
"""

from __future__ import annotations

import argparse
import json
import time

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    NonResonant,
    RealImag,
    enable_x64,
    generate_signal_toy,
)


enable_x64()


def make_model() -> DecayModel:
    return DecayModel(
        DecayChannel("B+", ("K+", "pi+", "pi-")),
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=64,
        normalization_pair=(0, 2),
    )


def generate(model, events: int, *, include_momenta: bool, seed: int):
    start = time.perf_counter()
    sample = generate_signal_toy(
        model,
        events,
        seed=seed,
        inverse_resolution=128,
        include_momenta=include_momenta,
    )
    elapsed = time.perf_counter() - start
    return sample, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=260905)
    args = parser.parse_args()
    if args.events < 1:
        raise ValueError("events must be positive")

    model = make_model()
    full, full_seconds = generate(
        model,
        args.events,
        include_momenta=True,
        seed=args.seed,
    )
    compact, compact_seconds = generate(
        model,
        args.events,
        include_momenta=False,
        seed=args.seed,
    )

    if not (
        bool((full.s12 == compact.s12).all())
        and bool((full.s13 == compact.s13).all())
        and bool((full.s23 == compact.s23).all())
    ):
        raise RuntimeError("compact generation changed Dalitz invariants")

    payload = {
        "events": args.events,
        "full_bytes": full.nbytes,
        "compact_bytes": compact.nbytes,
        "saved_bytes": full.nbytes - compact.nbytes,
        "retained_memory_reduction_fraction": 1.0 - compact.nbytes / full.nbytes,
        "full_generation_seconds": full_seconds,
        "compact_generation_seconds": compact_seconds,
        "full_has_momenta": full.p1 is not None,
        "compact_has_momenta": compact.p1 is not None,
    }
    print("MEMORY_BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
