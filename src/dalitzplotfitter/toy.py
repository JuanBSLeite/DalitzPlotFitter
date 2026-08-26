"""Toy Monte Carlo generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from .kinematics import PhaseSpaceSample, ThreeBodyPhaseSpace


def _concatenate_samples(samples: list[PhaseSpaceSample]) -> PhaseSpaceSample:
    """Concatenate accepted phase-space chunks into one unweighted sample."""

    if not samples:
        raise ValueError("At least one phase-space sample is required")

    def concatenate_optional(name: str):
        values = [getattr(sample, name) for sample in samples]
        if any(value is None for value in values):
            return None
        return jnp.concatenate(values, axis=0)

    s12 = jnp.concatenate([sample.s12 for sample in samples], axis=0)
    s13 = jnp.concatenate([sample.s13 for sample in samples], axis=0)
    s23 = jnp.concatenate([sample.s23 for sample in samples], axis=0)
    return PhaseSpaceSample(
        s12=s12,
        s13=s13,
        s23=s23,
        # Accepted events are unweighted. The proposal Jacobian has already
        # entered the accept-reject probability.
        weights=jnp.ones(s12.shape[0]),
        p1=concatenate_optional("p1"),
        p2=concatenate_optional("p2"),
        p3=concatenate_optional("p3"),
    )


@dataclass(frozen=True)
class ToyGenerator:
    """Generate unweighted toy events with accept-reject sampling.

    ``ThreeBodyPhaseSpace.generate`` uses a convenient proposal that is uniform
    in ``s12`` and conditionally uniform in ``s23``. Its returned ``weights`` are
    therefore the proposal-to-Dalitz-measure Jacobian. For a target intensity
    ``I(x)``, accept-reject uses the score

    ``score(x) = phase_space_weight(x) * I(x)``.

    ``pool_size`` is retained for backwards compatibility, but now denotes only
    the pilot sample used to estimate the accept-reject envelope. Accepted events
    are generated from fresh phase-space batches; they are not resampled from the
    pilot and therefore do not inherit finite-pool duplicate structure.

    Because a generic user-supplied intensity has no analytic global bound, the
    pilot maximum is multiplied by ``envelope_safety``. If a later batch exceeds
    the current envelope, every previously accepted event is discarded and the
    generation restarts with the enlarged envelope. This prevents an already
    observed envelope violation from biasing the returned sample.
    """

    phase_space: ThreeBodyPhaseSpace
    transformer: object
    pool_size: int = 200_000
    batch_size: int = 20_000
    envelope_safety: float = 1.2
    max_batches: int = 10_000

    def _score(
        self,
        sample: PhaseSpaceSample,
        data: Mapping[str, Array],
        intensity,
        parameters: Mapping[str, object],
    ) -> Array:
        values = jnp.asarray(intensity(data, parameters))
        if values.shape != sample.weights.shape:
            raise ValueError(
                "Toy intensity must return one value per phase-space event"
            )
        score = jnp.asarray(sample.weights) * jnp.clip(values, min=0.0)
        if not bool(jnp.all(jnp.isfinite(score))):
            raise ValueError("Toy intensity produced non-finite accept-reject weights")
        return score

    def generate(
        self,
        key: Array,
        size: int,
        intensity,
        parameters: Mapping[str, object],
    ) -> tuple[PhaseSpaceSample, dict[str, Array]]:
        if size <= 0:
            raise ValueError("size must be positive")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.envelope_safety <= 1.0:
            raise ValueError("envelope_safety must be greater than one")
        if self.max_batches <= 0:
            raise ValueError("max_batches must be positive")

        key_pilot, key_stream = jax.random.split(key)
        pilot = self.phase_space.generate(key_pilot, self.pool_size)
        pilot_data = self.transformer(pilot.as_momentum_dict())
        pilot_score = self._score(pilot, pilot_data, intensity, parameters)
        pilot_max = float(jnp.max(pilot_score))
        if not jnp.isfinite(pilot_max) or pilot_max <= 0.0:
            raise ValueError("Toy intensity has non-positive or non-finite integral")
        envelope = self.envelope_safety * pilot_max

        accepted_samples: list[PhaseSpaceSample] = []
        accepted_data: dict[str, list[Array]] = {}
        accepted_count = 0

        for _ in range(self.max_batches):
            if accepted_count >= size:
                break

            key_stream, key_candidates, key_uniform = jax.random.split(key_stream, 3)
            candidates = self.phase_space.generate(key_candidates, self.batch_size)
            data = self.transformer(candidates.as_momentum_dict())
            score = self._score(candidates, data, intensity, parameters)
            batch_max = float(jnp.max(score))

            if batch_max > envelope:
                # The sampled envelope was too small. Enlarge it and restart so
                # previously accepted events are not kept with the wrong
                # acceptance probability.
                envelope = self.envelope_safety * batch_max
                accepted_samples.clear()
                accepted_data.clear()
                accepted_count = 0
                continue

            uniforms = jax.random.uniform(key_uniform, shape=score.shape)
            accepted_indices = jnp.nonzero(
                uniforms < score / envelope,
                size=self.batch_size,
                fill_value=-1,
            )[0]
            n_accepted = int(jnp.sum(accepted_indices >= 0))
            if n_accepted == 0:
                continue

            n_needed = size - accepted_count
            indices = accepted_indices[: min(n_accepted, n_needed)]
            accepted_samples.append(candidates.take(indices))
            for name, value in data.items():
                accepted_data.setdefault(name, []).append(jnp.asarray(value)[indices])
            accepted_count += int(indices.shape[0])
        else:
            raise RuntimeError(
                "Accept-reject toy generation did not converge within max_batches"
            )

        if accepted_count < size:
            raise RuntimeError(
                "Accept-reject toy generation did not produce the requested sample"
            )

        selected = _concatenate_samples(accepted_samples).take(jnp.arange(size))
        selected_data = {
            name: jnp.concatenate(chunks, axis=0)[:size]
            for name, chunks in accepted_data.items()
        }
        return selected, selected_data
