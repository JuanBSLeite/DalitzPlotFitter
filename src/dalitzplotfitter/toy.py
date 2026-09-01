"""High-level pseudo-data generation from configured Dalitz models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.kinematics import PhaseSpaceSample
from dalitzplotfitter.sampling import weighted_resample


def _acceptance(efficiency, veto, data: dict[str, object]) -> jnp.ndarray:
    size = int(jnp.asarray(next(iter(data.values()))).shape[0])
    result = jnp.ones((size,), dtype=jnp.float64)
    if efficiency is not None:
        result = result * jnp.asarray(efficiency(data))
    if veto is not None:
        result = result * jnp.asarray(veto(data), dtype=result.dtype)
    return result


def _pool_size(size: int, pool_size: int | None) -> int:
    if size <= 0:
        raise ValueError("toy size must be positive")
    if pool_size is None:
        return max(100_000, 10 * size)
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    return int(pool_size)


def _merge_samples(samples: Sequence[PhaseSpaceSample]) -> PhaseSpaceSample:
    samples = tuple(sample for sample in samples if sample.size > 0)
    if not samples:
        raise ValueError("cannot merge an empty toy sample")

    have_momenta = [
        sample.p1 is not None and sample.p2 is not None and sample.p3 is not None
        for sample in samples
    ]
    if any(have_momenta) and not all(have_momenta):
        raise ValueError("all merged samples must consistently contain four-momenta")

    def concat(name: str):
        arrays = [getattr(sample, name) for sample in samples]
        if arrays[0] is None:
            return None
        return jnp.concatenate(arrays, axis=0)

    total = sum(sample.size for sample in samples)
    return PhaseSpaceSample(
        s12=concat("s12"),
        s13=concat("s13"),
        s23=concat("s23"),
        weights=jnp.ones((total,), dtype=samples[0].s12.dtype),
        p1=concat("p1"),
        p2=concat("p2"),
        p3=concat("p3"),
    )


@dataclass(frozen=True)
class ToyBackground:
    """One background component for high-level toy generation.

    For multiple backgrounds, the first ``N-1`` entries define relative
    fractions and the final entry is the remainder, matching ``BackgroundSpec``.
    """

    name: str
    shape: object
    fraction: float | None = None
    apply_veto: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("toy background name must be non-empty")
        if not callable(self.shape):
            raise TypeError("toy background shape must be callable")
        if self.fraction is not None and not 0.0 <= float(self.fraction) <= 1.0:
            raise ValueError("toy background fraction must lie in [0, 1]")


@dataclass(frozen=True)
class CPToyBackground:
    """One charge-aware background component for CP toy generation."""

    name: str
    plus_shape: object
    minus_shape: object | None = None
    fraction: float | None = None
    apply_veto: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CP toy background name must be non-empty")
        if not callable(self.plus_shape):
            raise TypeError("CP toy plus_shape must be callable")
        if self.minus_shape is not None and not callable(self.minus_shape):
            raise TypeError("CP toy minus_shape must be callable")
        if self.fraction is not None and not 0.0 <= float(self.fraction) <= 1.0:
            raise ValueError("CP toy background fraction must lie in [0, 1]")

    @property
    def resolved_minus_shape(self):
        return self.plus_shape if self.minus_shape is None else self.minus_shape


def _background_weights(backgrounds: Sequence[object]) -> np.ndarray:
    backgrounds = tuple(backgrounds)
    n = len(backgrounds)
    if n == 0:
        return np.empty((0,), dtype=float)
    if n == 1:
        if getattr(backgrounds[0], "fraction") is not None:
            raise ValueError("a single toy background does not need a relative fraction")
        return np.ones((1,), dtype=float)
    explicit = []
    for background in backgrounds[:-1]:
        if getattr(background, "fraction") is None:
            raise ValueError("all toy backgrounds except the last require a relative fraction")
        explicit.append(float(background.fraction))
    if getattr(backgrounds[-1], "fraction") is not None:
        raise ValueError("the last toy background is the remainder and must not define fraction")
    remainder = 1.0 - sum(explicit)
    weights = np.asarray(explicit + [remainder], dtype=float)
    if np.any(weights < 0.0):
        raise ValueError("toy background fractions sum to more than one")
    return weights


def generate_signal_toy(
    model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    efficiency=None,
    veto=None,
    seed: int | None = None,
    pool_size: int | None = None,
) -> PhaseSpaceSample:
    """Generate unweighted signal pseudo-data from an already configured model."""

    n_pool = _pool_size(size, pool_size)
    pool = model.generate_phase_space(n_pool, seed=seed)
    data = pool.as_dict()
    values = {} if parameters is None else parameters
    weights = (
        jnp.asarray(pool.weights)
        * _acceptance(efficiency, veto, data)
        * jnp.asarray(model.intensity(data, values))
    )
    key = jax.random.key(0 if seed is None else int(seed) + 1)
    return weighted_resample(key, pool, weights, size, replace=True)


def generate_toy(
    model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    efficiency=None,
    veto=None,
    signal_fraction: float = 1.0,
    backgrounds: Sequence[ToyBackground] = (),
    seed: int | None = None,
    pool_size: int | None = None,
    shuffle: bool = True,
) -> PhaseSpaceSample:
    """Generate signal or signal+background pseudo-data in one call.

    ``signal_fraction`` is the total signal fraction. Background ``fraction``
    values describe the relative composition of the total background, with the
    final background acting as the remainder.
    """

    if not 0.0 <= float(signal_fraction) <= 1.0:
        raise ValueError("signal_fraction must lie in [0, 1]")
    backgrounds = tuple(backgrounds)
    if float(signal_fraction) < 1.0 and not backgrounds:
        raise ValueError("signal_fraction < 1 requires at least one background")
    if backgrounds and float(signal_fraction) >= 1.0:
        raise ValueError("backgrounds require signal_fraction < 1")

    n_pool = _pool_size(size, pool_size)
    rng = np.random.default_rng(seed)
    n_signal = int(rng.binomial(size, float(signal_fraction))) if backgrounds else size
    n_background = size - n_signal
    bg_weights = _background_weights(backgrounds)
    bg_counts = (
        rng.multinomial(n_background, bg_weights)
        if n_background > 0
        else np.zeros(len(backgrounds), dtype=int)
    )

    pool = model.generate_phase_space(n_pool, seed=seed)
    data = pool.as_dict()
    values = {} if parameters is None else parameters
    accepted = _acceptance(efficiency, veto, data)
    samples: list[PhaseSpaceSample] = []

    if n_signal > 0:
        signal_weights = (
            jnp.asarray(pool.weights)
            * accepted
            * jnp.asarray(model.intensity(data, values))
        )
        samples.append(
            weighted_resample(
                jax.random.key(1 if seed is None else int(seed) + 1),
                pool,
                signal_weights,
                n_signal,
                replace=True,
            )
        )

    for index, (background, count) in enumerate(zip(backgrounds, bg_counts)):
        if int(count) == 0:
            continue
        background_weights = jnp.asarray(pool.weights) * jnp.asarray(background.shape(data))
        if veto is not None and background.apply_veto:
            background_weights = background_weights * jnp.asarray(veto(data))
        samples.append(
            weighted_resample(
                jax.random.key((10 if seed is None else int(seed) + 10) + index),
                pool,
                background_weights,
                int(count),
                replace=True,
            )
        )

    toy = _merge_samples(samples)
    if shuffle and toy.size > 1:
        indices = jax.random.permutation(
            jax.random.key(100 if seed is None else int(seed) + 100), toy.size
        )
        toy = toy.take(indices)
        toy = PhaseSpaceSample(
            s12=toy.s12,
            s13=toy.s13,
            s23=toy.s23,
            weights=jnp.ones((toy.size,), dtype=toy.s12.dtype),
            p1=toy.p1,
            p2=toy.p2,
            p3=toy.p3,
        )
    return toy


def _integral(model, parameters, efficiency, veto) -> float:
    sample = model.normalization_sample
    data = sample.as_dict()
    value = jnp.mean(
        jnp.asarray(sample.weights)
        * _acceptance(efficiency, veto, data)
        * jnp.asarray(model.intensity(data, parameters))
    )
    return float(value)


def generate_cp_toy(
    plus_model,
    minus_model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    plus_efficiency=None,
    minus_efficiency=None,
    plus_veto=None,
    minus_veto=None,
    signal_fraction: float = 1.0,
    backgrounds: Sequence[CPToyBackground] = (),
    seed: int | None = None,
    pool_size: int | None = None,
    shuffle: bool = True,
) -> tuple[PhaseSpaceSample, PhaseSpaceSample]:
    """Generate a simultaneous ``(B+, B-)`` pseudoexperiment.

    Signal and background charge splits are computed from their deterministic
    accepted integrals, so the generated charge asymmetry follows the same joint
    normalization convention used by ``CPJointNLL``.
    """

    if not 0.0 <= float(signal_fraction) <= 1.0:
        raise ValueError("signal_fraction must lie in [0, 1]")
    backgrounds = tuple(backgrounds)
    if float(signal_fraction) < 1.0 and not backgrounds:
        raise ValueError("signal_fraction < 1 requires at least one CP background")
    if backgrounds and float(signal_fraction) >= 1.0:
        raise ValueError("CP backgrounds require signal_fraction < 1")

    values = {} if parameters is None else parameters
    n_pool = _pool_size(size, pool_size)
    rng = np.random.default_rng(seed)
    n_signal = int(rng.binomial(size, float(signal_fraction))) if backgrounds else size
    n_background = size - n_signal

    i_plus = _integral(plus_model, values, plus_efficiency, plus_veto)
    i_minus = _integral(minus_model, values, minus_efficiency, minus_veto)
    signal_plus_probability = i_plus / (i_plus + i_minus)
    n_signal_plus = int(rng.binomial(n_signal, signal_plus_probability))
    n_signal_minus = n_signal - n_signal_plus

    bg_mix = _background_weights(backgrounds)
    bg_counts = (
        rng.multinomial(n_background, bg_mix)
        if n_background > 0
        else np.zeros(len(backgrounds), dtype=int)
    )

    plus_pool = plus_model.generate_phase_space(n_pool, seed=seed)
    minus_pool = minus_model.generate_phase_space(
        n_pool, seed=None if seed is None else int(seed) + 1
    )
    plus_data = plus_pool.as_dict()
    minus_data = minus_pool.as_dict()
    plus_samples: list[PhaseSpaceSample] = []
    minus_samples: list[PhaseSpaceSample] = []

    if n_signal_plus > 0:
        plus_signal_weights = (
            jnp.asarray(plus_pool.weights)
            * _acceptance(plus_efficiency, plus_veto, plus_data)
            * jnp.asarray(plus_model.intensity(plus_data, values))
        )
        plus_samples.append(
            weighted_resample(
                jax.random.key(1 if seed is None else int(seed) + 2),
                plus_pool,
                plus_signal_weights,
                n_signal_plus,
                replace=True,
            )
        )
    if n_signal_minus > 0:
        minus_signal_weights = (
            jnp.asarray(minus_pool.weights)
            * _acceptance(minus_efficiency, minus_veto, minus_data)
            * jnp.asarray(minus_model.intensity(minus_data, values))
        )
        minus_samples.append(
            weighted_resample(
                jax.random.key(2 if seed is None else int(seed) + 3),
                minus_pool,
                minus_signal_weights,
                n_signal_minus,
                replace=True,
            )
        )

    for index, (background, count) in enumerate(zip(backgrounds, bg_counts)):
        if int(count) == 0:
            continue
        plus_shape = jnp.asarray(background.plus_shape(plus_data))
        minus_shape = jnp.asarray(background.resolved_minus_shape(minus_data))
        plus_norm_sample = plus_model.normalization_sample
        minus_norm_sample = minus_model.normalization_sample
        plus_norm_data = plus_norm_sample.as_dict()
        minus_norm_data = minus_norm_sample.as_dict()
        j_plus_values = jnp.asarray(background.plus_shape(plus_norm_data))
        j_minus_values = jnp.asarray(background.resolved_minus_shape(minus_norm_data))
        if background.apply_veto:
            if plus_veto is not None:
                plus_shape = plus_shape * jnp.asarray(plus_veto(plus_data))
                j_plus_values = j_plus_values * jnp.asarray(plus_veto(plus_norm_data))
            if minus_veto is not None:
                minus_shape = minus_shape * jnp.asarray(minus_veto(minus_data))
                j_minus_values = j_minus_values * jnp.asarray(minus_veto(minus_norm_data))
        j_plus = float(jnp.mean(jnp.asarray(plus_norm_sample.weights) * j_plus_values))
        j_minus = float(jnp.mean(jnp.asarray(minus_norm_sample.weights) * j_minus_values))
        plus_probability = j_plus / (j_plus + j_minus)
        count_plus = int(rng.binomial(int(count), plus_probability))
        count_minus = int(count) - count_plus
        if count_plus > 0:
            plus_samples.append(
                weighted_resample(
                    jax.random.key((20 if seed is None else int(seed) + 20) + index),
                    plus_pool,
                    jnp.asarray(plus_pool.weights) * plus_shape,
                    count_plus,
                    replace=True,
                )
            )
        if count_minus > 0:
            minus_samples.append(
                weighted_resample(
                    jax.random.key((40 if seed is None else int(seed) + 40) + index),
                    minus_pool,
                    jnp.asarray(minus_pool.weights) * minus_shape,
                    count_minus,
                    replace=True,
                )
            )

    def finish(samples: list[PhaseSpaceSample], key_seed: int) -> PhaseSpaceSample:
        if not samples:
            empty_pool = plus_pool if key_seed == 0 else minus_pool
            return PhaseSpaceSample(
                s12=empty_pool.s12[:0],
                s13=empty_pool.s13[:0],
                s23=empty_pool.s23[:0],
                weights=jnp.ones((0,), dtype=empty_pool.s12.dtype),
                p1=None if empty_pool.p1 is None else empty_pool.p1[:0],
                p2=None if empty_pool.p2 is None else empty_pool.p2[:0],
                p3=None if empty_pool.p3 is None else empty_pool.p3[:0],
            )
        toy = _merge_samples(samples)
        if shuffle and toy.size > 1:
            toy = toy.take(
                jax.random.permutation(
                    jax.random.key(
                        (200 if seed is None else int(seed) + 200) + key_seed
                    ),
                    toy.size,
                )
            )
            toy = PhaseSpaceSample(
                s12=toy.s12,
                s13=toy.s13,
                s23=toy.s23,
                weights=jnp.ones((toy.size,), dtype=toy.s12.dtype),
                p1=toy.p1,
                p2=toy.p2,
                p3=toy.p3,
            )
        return toy

    plus_toy = finish(plus_samples, 0)
    minus_toy = finish(minus_samples, 1)
    if plus_toy.size + minus_toy.size != size:
        raise RuntimeError("internal CP toy generation count mismatch")
    return plus_toy, minus_toy


__all__ = [
    "CPToyBackground",
    "ToyBackground",
    "generate_cp_toy",
    "generate_signal_toy",
    "generate_toy",
]
