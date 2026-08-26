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

    ``ThreeBodyPhaseSpace.generate`` samples uniformly in the unit square that
    is mapped onto the Dalitz plot. Its returned ``weights`` are the Jacobian of
    that transformation. Therefore the accept-reject target relative to the
    unit-square proposal is

    ``score(u) = phase_space_weight(u) * I(x(u))``.

    Before any toy event is generated, :meth:`estimate_maximum` searches this
    score deterministically: first on a global unit-square grid, then through
    local refinements around several of the best grid points. The final envelope
    is ``envelope_safety`` times that maximum. During generation, any observed
    violation still causes a complete restart with an enlarged envelope.

    ``pool_size`` is retained only for backwards API compatibility with older
    examples. Envelope estimation no longer relies on a random pilot sample.
    """

    phase_space: ThreeBodyPhaseSpace
    transformer: object
    pool_size: int = 200_000
    batch_size: int = 20_000
    envelope_safety: float = 1.2
    envelope_grid_size: int = 160
    envelope_refinement_size: int = 17
    envelope_refinement_levels: int = 4
    envelope_top_k: int = 12
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

    def _score_unit_points(
        self,
        unit_points: Array,
        intensity,
        parameters: Mapping[str, object],
    ) -> Array:
        sample = self.phase_space.from_unit_square(unit_points)
        data = self.transformer(sample.as_momentum_dict())
        return self._score(sample, data, intensity, parameters)

    def estimate_maximum(
        self,
        intensity,
        parameters: Mapping[str, object],
    ) -> tuple[float, Array]:
        """Estimate the global accept-reject score maximum deterministically.

        The search starts from a regular grid of cell centres over ``[0, 1]^2``.
        Several best candidates are retained and independently refined. Using
        cell centres avoids evaluating exactly on kinematic boundaries, where
        helicity coordinates can become singular while the physical phase-space
        measure vanishes.
        """

        if self.envelope_grid_size < 2:
            raise ValueError("envelope_grid_size must be at least two")
        if self.envelope_refinement_size < 3:
            raise ValueError("envelope_refinement_size must be at least three")
        if self.envelope_refinement_levels < 0:
            raise ValueError("envelope_refinement_levels must be non-negative")
        if self.envelope_top_k <= 0:
            raise ValueError("envelope_top_k must be positive")

        grid_size = self.envelope_grid_size
        centres = (jnp.arange(grid_size, dtype=float) + 0.5) / grid_size
        u1, u2 = jnp.meshgrid(centres, centres, indexing="ij")
        points = jnp.stack([u1.ravel(), u2.ravel()], axis=1)
        scores = self._score_unit_points(points, intensity, parameters)

        top_k = min(self.envelope_top_k, int(points.shape[0]))
        top_indices = jnp.argsort(scores)[-top_k:]
        best_points = points[top_indices]
        best_scores = scores[top_indices]

        # Half-width of the global-grid cell around each retained candidate.
        radius = 0.5 / grid_size
        refinement_size = self.envelope_refinement_size
        for _ in range(self.envelope_refinement_levels):
            offsets_1d = jnp.linspace(-radius, radius, refinement_size)
            du1, du2 = jnp.meshgrid(offsets_1d, offsets_1d, indexing="ij")
            offsets = jnp.stack([du1.ravel(), du2.ravel()], axis=1)
            local_points = best_points[:, None, :] + offsets[None, :, :]

            # Stay strictly inside the unit square to avoid boundary coordinate
            # singularities. The excluded strip shrinks far below the final
            # refinement resolution and carries vanishing phase-space measure.
            eps = jnp.finfo(local_points.dtype).eps * 16
            local_points = jnp.clip(local_points, eps, 1.0 - eps)
            local_points = local_points.reshape((-1, 2))
            local_scores = self._score_unit_points(
                local_points,
                intensity,
                parameters,
            )

            combined_points = jnp.concatenate([best_points, local_points], axis=0)
            combined_scores = jnp.concatenate([best_scores, local_scores], axis=0)
            top_indices = jnp.argsort(combined_scores)[-top_k:]
            best_points = combined_points[top_indices]
            best_scores = combined_scores[top_indices]

            # Adjacent refinement points are separated by this amount. Search
            # only that smaller neighbourhood on the next level.
            radius = 2.0 * radius / (refinement_size - 1)

        maximum_index = int(jnp.argmax(best_scores))
        maximum = float(best_scores[maximum_index])
        maximum_point = best_points[maximum_index]
        if not jnp.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("Toy intensity has non-positive or non-finite integral")
        return maximum, maximum_point

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

        maximum, _ = self.estimate_maximum(intensity, parameters)
        envelope = self.envelope_safety * maximum
        key_stream = key

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
                # The deterministic search missed a larger value. Enlarge the
                # envelope and restart so no previously accepted event keeps an
                # inconsistent acceptance probability.
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
