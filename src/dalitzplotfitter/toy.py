"""Toy Monte Carlo generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from .kinematics import PhaseSpaceSample, ThreeBodyPhaseSpace


@dataclass(frozen=True)
class ToyGenerator:
    """Generate unweighted toy events by resampling a phase-space pool.

    The pool is sampled independently from the normalization sample used in a fit.
    Selection probabilities are proportional to the phase-space Jacobian times the
    supplied intensity, so the returned events follow the requested Dalitz density.
    """

    phase_space: ThreeBodyPhaseSpace
    transformer: object
    pool_size: int = 200_000

    def generate(
        self,
        key: Array,
        size: int,
        intensity,
        parameters: Mapping[str, object],
    ) -> tuple[PhaseSpaceSample, dict[str, Array]]:
        if size <= 0:
            raise ValueError("size must be positive")
        if self.pool_size < size:
            raise ValueError("pool_size must be at least as large as the toy size")

        key_pool, key_choice = jax.random.split(key)
        pool = self.phase_space.generate(key_pool, self.pool_size)
        data = self.transformer(pool.as_momentum_dict())
        values = jnp.asarray(intensity(data, parameters))
        probabilities = jnp.asarray(pool.weights) * jnp.clip(values, min=0.0)
        total = jnp.sum(probabilities)
        if not jnp.isfinite(total) or total <= 0.0:
            raise ValueError("Toy intensity has non-positive or non-finite integral")
        probabilities = probabilities / total
        indices = jax.random.choice(
            key_choice,
            self.pool_size,
            shape=(size,),
            replace=True,
            p=probabilities,
        )
        selected = pool.take(indices)
        selected_data = {name: value[indices] for name, value in data.items()}
        return selected, selected_data
