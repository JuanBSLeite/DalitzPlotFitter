"""Utilities for turning weighted Monte Carlo into unweighted pseudo-data."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.kinematics import PhaseSpaceSample


def weighted_resample(
    key: Array,
    sample: PhaseSpaceSample,
    weights: Array,
    size: int,
    *,
    replace: bool = True,
) -> PhaseSpaceSample:
    """Draw unweighted events from a weighted phase-space sample.

    ``weights`` should contain the complete target importance weight for each
    candidate, typically

    ``w_target = w_PS * |A(x)|^2``.

    The returned sample has unit event weights because it represents unweighted
    pseudo-data distributed according to the discrete weighted candidate pool.
    For closure studies the candidate pool should be substantially larger than
    the requested pseudo-data sample so finite-pool duplication effects are
    negligible.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    weights = jnp.asarray(weights)
    if weights.shape != sample.weights.shape:
        raise ValueError("weights must have the same shape as sample.weights")
    if not bool(jnp.all(jnp.isfinite(weights))):
        raise ValueError("weights must be finite")
    if bool(jnp.any(weights < 0.0)):
        raise ValueError("weights must be non-negative")
    total = jnp.sum(weights)
    if not bool(jnp.isfinite(total)) or float(total) <= 0.0:
        raise ValueError("weights must have a positive finite sum")
    if not replace and size > sample.size:
        raise ValueError("cannot sample more events than candidates without replacement")

    probabilities = weights / total
    indices = jax.random.choice(
        key,
        sample.size,
        shape=(size,),
        replace=replace,
        p=probabilities,
    )
    selected = sample.take(indices)
    return PhaseSpaceSample(
        s12=selected.s12,
        s13=selected.s13,
        s23=selected.s23,
        weights=jnp.ones(size, dtype=selected.s12.dtype),
        p1=selected.p1,
        p2=selected.p2,
        p3=selected.p3,
    )
