"""Benchmark dense and sparse Square-Dalitz SCF migration.

Example
-------
python benchmarks/benchmark_scf_migration.py --bins-mprime 40 --bins-thetaprime 40 --neighbors 9

The synthetic migration sends every true bin to a small local set of
reconstructed bins and is row-normalized.  The benchmark reports storage size,
JIT compile+first-call time and steady-state migration time on the active JAX
device.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from dalitzplotfitter import SparseMigration, SquareDalitzSCFMap, enable_x64


enable_x64()


def _local_sparse_migration(n_bins: int, neighbors: int) -> SparseMigration:
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    offsets = np.arange(neighbors, dtype=np.int32) - neighbors // 2
    true = np.repeat(np.arange(n_bins, dtype=np.int32), neighbors)
    reco = (true + np.tile(offsets, n_bins)) % n_bins
    probability = np.full(true.size, 1.0 / neighbors, dtype=np.float64)
    return SparseMigration(
        true_indices=jnp.asarray(true),
        reco_indices=jnp.asarray(reco),
        probabilities=jnp.asarray(probability),
        n_bins=n_bins,
    )


def _seconds(function):
    start = time.perf_counter()
    value = function()
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value, time.perf_counter() - start


def _benchmark(function, density, repeats: int):
    compiled = jax.jit(function)
    _, first = _seconds(lambda: compiled(density))
    times = []
    for index in range(repeats):
        shifted = density * (1.0 + 1e-8 * (index + 1))
        _, elapsed = _seconds(lambda shifted=shifted: compiled(shifted))
        times.append(elapsed)
    return first, times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins-mprime", type=int, default=40)
    parser.add_argument("--bins-thetaprime", type=int, default=40)
    parser.add_argument("--neighbors", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    n_bins = args.bins_mprime * args.bins_thetaprime
    sparse = _local_sparse_migration(n_bins, args.neighbors)
    dense_matrix = sparse.to_dense()
    fraction = jnp.full((n_bins,), 0.2, dtype=jnp.float64)
    common = dict(
        scf_fraction=fraction,
        mother_mass=5.27934,
        masses=(0.493677, 0.13957039, 0.13957039),
        bins_mprime=args.bins_mprime,
        bins_thetaprime=args.bins_thetaprime,
        pair=(0, 2),
    )
    dense = SquareDalitzSCFMap(migration=dense_matrix, storage="dense", **common)
    sparse_map = SquareDalitzSCFMap(migration=sparse, **common)
    density = 1.0 + 0.1 * jnp.arange(n_bins, dtype=jnp.float64) / max(n_bins, 1)

    dense_first, dense_times = _benchmark(
        dense.smeared_bin_density, density, args.repeats
    )
    sparse_first, sparse_times = _benchmark(
        sparse_map.smeared_bin_density, density, args.repeats
    )

    dense_bytes = n_bins * n_bins * 8
    sparse_bytes = sparse.nnz * (4 + 4 + 8)
    payload = {
        "device": str(jax.devices()[0]),
        "platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "bins_mprime": args.bins_mprime,
        "bins_thetaprime": args.bins_thetaprime,
        "n_bins": n_bins,
        "neighbors_per_true_bin": args.neighbors,
        "nnz": sparse.nnz,
        "migration_density": sparse.density,
        "dense_storage_bytes": dense_bytes,
        "sparse_storage_bytes_approx": sparse_bytes,
        "storage_reduction_factor": dense_bytes / max(sparse_bytes, 1),
        "dense_first_seconds": dense_first,
        "sparse_first_seconds": sparse_first,
        "dense_steady_seconds_mean": float(np.mean(dense_times)),
        "sparse_steady_seconds_mean": float(np.mean(sparse_times)),
        "dense_steady_seconds_min": float(np.min(dense_times)),
        "sparse_steady_seconds_min": float(np.min(sparse_times)),
        "steady_speedup_sparse_over_dense": float(
            np.mean(dense_times) / np.mean(sparse_times)
        ),
    }
    print("SCF_BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
