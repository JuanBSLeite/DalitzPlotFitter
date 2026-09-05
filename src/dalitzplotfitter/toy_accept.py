"""Laura++-style accept-reject pseudo-data generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.amplitude.components import coefficient_value
from dalitzplotfitter.kinematics import PhaseSpaceMC, PhaseSpaceSample


def _acceptance(efficiency, veto, data: dict[str, object]) -> jnp.ndarray:
    size = int(jnp.asarray(next(iter(data.values()))).shape[0])
    result = jnp.ones((size,), dtype=jnp.float64)
    if efficiency is not None:
        result = result * jnp.asarray(efficiency(data))
    if veto is not None:
        result = result * jnp.asarray(veto(data), dtype=result.dtype)
    return result


def _pilot_size(size: int, pool_size: int | None) -> int:
    if size <= 0:
        raise ValueError("toy size must be positive")
    if pool_size is not None:
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        return int(pool_size)
    return max(20_000, min(100_000, 2 * size))


def _derived_seed(seed: int | None, offset: int) -> int | None:
    if seed is None:
        return None
    return (int(seed) + int(offset)) % (2**32)


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


def _empty_sample(
    model,
    seed: int | None = None,
    *,
    include_momenta: bool = True,
) -> PhaseSpaceSample:
    sample = model.generate_phase_space(
        1,
        seed=seed,
        include_momenta=include_momenta,
    )
    return PhaseSpaceSample(
        s12=sample.s12[:0],
        s13=sample.s13[:0],
        s23=sample.s23[:0],
        weights=jnp.ones((0,), dtype=sample.s12.dtype),
        p1=None if sample.p1 is None else sample.p1[:0],
        p2=None if sample.p2 is None else sample.p2[:0],
        p3=None if sample.p3 is None else sample.p3[:0],
    )


def _scores(pool: PhaseSpaceSample, density) -> jax.Array:
    values = jnp.asarray(pool.weights) * jnp.asarray(density)
    if values.shape != (pool.size,):
        raise ValueError(
            f"toy density must return one value per event, got {values.shape} for {pool.size} events"
        )
    valid = jnp.all(jnp.isfinite(values)) & jnp.all(values >= 0.0)
    if not bool(jax.device_get(valid)):
        raise ValueError("toy generation weights must be finite and non-negative")
    return values


def _frozen_model_intensity(model, values: Mapping[str, object]):
    """Build one JIT intensity with component scales frozen at the toy truth.

    Calling DecayModel.intensity directly inside every proposal batch also
    recomputes component-normalization integrals.  Toy truth parameters are
    fixed, so those scales and complex coefficients can be resolved once.
    """

    components = tuple(model.amplitude_model.components)
    resolved = []
    for component in components:
        scale = jnp.asarray(model._component_scale(component, values))
        coefficient = jnp.asarray(coefficient_value(component.coefficient, values))
        jax.block_until_ready(scale)
        jax.block_until_ready(coefficient)
        resolved.append((component, scale, coefficient))

    def intensity(data):
        total = None
        for component, scale, coefficient in resolved:
            dynamics = jnp.asarray(component.function(data, values))
            term = coefficient * scale * dynamics
            total = term if total is None else total + term
        total = jnp.asarray(total)
        return jnp.real(total * jnp.conj(total))

    return jax.jit(intensity)


def _attach_momenta(model, sample: PhaseSpaceSample, seed: int | None) -> PhaseSpaceSample:
    if sample.p1 is not None:
        return sample
    generator = PhaseSpaceMC(
        model.channel.parent_mass,
        model.channel.daughter_masses,
    )
    return generator.attach_momenta(sample, seed=seed)


def _accept_reject_component(
    model,
    size: int,
    density_function,
    *,
    seed: int | None,
    pool_size: int | None,
    batch_size: int | None,
    envelope_safety: float,
    max_restarts: int,
) -> PhaseSpaceSample:
    """Generate one unweighted component with a monitored global envelope."""

    if size <= 0:
        raise ValueError("toy size must be positive")
    if envelope_safety <= 1.0:
        raise ValueError("envelope_safety must be greater than one")
    if max_restarts < 0:
        raise ValueError("max_restarts must be non-negative")

    compiled_density = jax.jit(density_function)
    n_pilot = _pilot_size(size, pool_size)

    # Built-in amplitudes need only Dalitz invariants.  Try the compact proposal
    # first and fall back to full four-vectors only for custom densities that
    # explicitly request momentum fields.
    compact_proposal = True
    pilot = model.generate_phase_space(
        n_pilot,
        seed=_derived_seed(seed, 1),
        include_momenta=False,
    )
    try:
        pilot_scores = _scores(pilot, compiled_density(pilot.as_dict()))
    except KeyError:
        compact_proposal = False
        pilot = model.generate_phase_space(
            n_pilot,
            seed=_derived_seed(seed, 1),
            include_momenta=True,
        )
        pilot_scores = _scores(pilot, compiled_density(pilot.as_dict()))

    observed_max_device, mean_score_device = jax.device_get(
        (jnp.max(pilot_scores), jnp.mean(pilot_scores))
    )
    observed_max = float(observed_max_device)
    if observed_max <= 0.0:
        raise ValueError("toy density is zero over the pilot phase-space sample")
    envelope = envelope_safety * observed_max

    estimated_efficiency = float(mean_score_device) / envelope
    estimated_efficiency = min(max(estimated_efficiency, 1e-4), 1.0)
    if batch_size is None:
        batch_size = int(
            min(
                500_000,
                max(4_096, np.ceil(1.25 * size / estimated_efficiency)),
            )
        )
    elif batch_size <= 0:
        raise ValueError("batch_size must be positive")
    else:
        batch_size = int(batch_size)

    accepted: list[PhaseSpaceSample] = []
    n_accepted = 0
    restarts = 0
    proposal_index = 0

    while n_accepted < size:
        proposal_index += 1
        if proposal_index > 10_000:
            raise RuntimeError("accept-reject toy generation did not converge")
        pool = model.generate_phase_space(
            batch_size,
            seed=_derived_seed(seed, 10_000 + proposal_index),
            include_momenta=not compact_proposal,
        )
        score = _scores(pool, compiled_density(pool.as_dict()))
        batch_max = float(jax.device_get(jnp.max(score)))
        if batch_max > envelope:
            restarts += 1
            if restarts > max_restarts:
                raise RuntimeError(
                    "accept-reject envelope was exceeded too many times; "
                    "increase pool_size or envelope_safety"
                )
            envelope = envelope_safety * batch_max
            accepted.clear()
            n_accepted = 0
            continue

        accept_key = jax.random.key(
            0
            if seed is None
            else _derived_seed(seed, 913_579 + proposal_index)
        )
        mask = jax.random.uniform(
            accept_key,
            (pool.size,),
            dtype=score.dtype,
        ) < (score / envelope)
        indices = np.flatnonzero(np.asarray(jax.device_get(mask), dtype=bool))
        if indices.size == 0:
            continue
        needed = size - n_accepted
        selected = indices[:needed]
        accepted.append(
            pool.take(jnp.asarray(selected, dtype=jnp.int32)).without_momenta()
        )
        n_accepted += int(selected.size)

    toy = _merge_samples(accepted)
    if toy.size != size:
        raise RuntimeError("internal accept-reject toy generation count mismatch")
    return toy


@dataclass(frozen=True)
class ToyBackground:
    """One background component for high-level toy generation."""

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
    if not backgrounds:
        return np.empty((0,), dtype=float)
    if len(backgrounds) == 1:
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
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    include_momenta: bool = True,
) -> PhaseSpaceSample:
    values = {} if parameters is None else parameters
    intensity = _frozen_model_intensity(model, values)

    def signal_density(data):
        return _acceptance(efficiency, veto, data) * intensity(data)

    toy = _accept_reject_component(
        model,
        size,
        signal_density,
        seed=seed,
        pool_size=pool_size,
        batch_size=batch_size,
        envelope_safety=envelope_safety,
        max_restarts=max_restarts,
    )
    if include_momenta:
        toy = _attach_momenta(model, toy, _derived_seed(seed, 900_001))
    return toy


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
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    include_momenta: bool = True,
) -> PhaseSpaceSample:
    if size <= 0:
        raise ValueError("toy size must be positive")
    if not 0.0 <= float(signal_fraction) <= 1.0:
        raise ValueError("signal_fraction must lie in [0, 1]")
    backgrounds = tuple(backgrounds)
    if float(signal_fraction) < 1.0 and not backgrounds:
        raise ValueError("signal_fraction < 1 requires at least one background")
    if backgrounds and float(signal_fraction) >= 1.0:
        raise ValueError("backgrounds require signal_fraction < 1")

    rng = np.random.default_rng(seed)
    n_signal = int(rng.binomial(size, float(signal_fraction))) if backgrounds else size
    n_background = size - n_signal
    bg_weights = _background_weights(backgrounds)
    bg_counts = (
        rng.multinomial(n_background, bg_weights)
        if n_background > 0
        else np.zeros(len(backgrounds), dtype=int)
    )
    values = {} if parameters is None else parameters
    samples: list[PhaseSpaceSample] = []

    if n_signal > 0:
        intensity = _frozen_model_intensity(model, values)

        def signal_density(data):
            return _acceptance(efficiency, veto, data) * intensity(data)

        samples.append(
            _accept_reject_component(
                model,
                n_signal,
                signal_density,
                seed=_derived_seed(seed, 1),
                pool_size=pool_size,
                batch_size=batch_size,
                envelope_safety=envelope_safety,
                max_restarts=max_restarts,
            )
        )

    for index, (background, count) in enumerate(zip(backgrounds, bg_counts)):
        if int(count) == 0:
            continue

        def background_density(data, background=background):
            result = jnp.asarray(background.shape(data))
            if veto is not None and background.apply_veto:
                result = result * jnp.asarray(veto(data))
            return result

        samples.append(
            _accept_reject_component(
                model,
                int(count),
                background_density,
                seed=_derived_seed(seed, 100 + index),
                pool_size=pool_size,
                batch_size=batch_size,
                envelope_safety=envelope_safety,
                max_restarts=max_restarts,
            )
        )

    toy = _merge_samples(samples)
    if shuffle and toy.size > 1:
        toy = toy.take(
            jax.random.permutation(
                jax.random.key(100 if seed is None else (int(seed) + 100) % (2**32)),
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
    if include_momenta:
        toy = _attach_momenta(model, toy, _derived_seed(seed, 900_002))
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
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    include_momenta: bool = True,
) -> tuple[PhaseSpaceSample, PhaseSpaceSample]:
    if size <= 0:
        raise ValueError("toy size must be positive")
    if not 0.0 <= float(signal_fraction) <= 1.0:
        raise ValueError("signal_fraction must lie in [0, 1]")
    backgrounds = tuple(backgrounds)
    if float(signal_fraction) < 1.0 and not backgrounds:
        raise ValueError("signal_fraction < 1 requires at least one CP background")
    if backgrounds and float(signal_fraction) >= 1.0:
        raise ValueError("CP backgrounds require signal_fraction < 1")

    values = {} if parameters is None else parameters
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
    plus_samples: list[PhaseSpaceSample] = []
    minus_samples: list[PhaseSpaceSample] = []

    if n_signal_plus > 0:
        plus_intensity = _frozen_model_intensity(plus_model, values)

        def plus_signal_density(data):
            return _acceptance(plus_efficiency, plus_veto, data) * plus_intensity(data)

        plus_samples.append(
            _accept_reject_component(
                plus_model,
                n_signal_plus,
                plus_signal_density,
                seed=_derived_seed(seed, 10),
                pool_size=pool_size,
                batch_size=batch_size,
                envelope_safety=envelope_safety,
                max_restarts=max_restarts,
            )
        )

    if n_signal_minus > 0:
        minus_intensity = _frozen_model_intensity(minus_model, values)

        def minus_signal_density(data):
            return _acceptance(minus_efficiency, minus_veto, data) * minus_intensity(data)

        minus_samples.append(
            _accept_reject_component(
                minus_model,
                n_signal_minus,
                minus_signal_density,
                seed=_derived_seed(seed, 20),
                pool_size=pool_size,
                batch_size=batch_size,
                envelope_safety=envelope_safety,
                max_restarts=max_restarts,
            )
        )

    for index, (background, count) in enumerate(zip(backgrounds, bg_counts)):
        if int(count) == 0:
            continue
        plus_norm_sample = plus_model.normalization_sample
        minus_norm_sample = minus_model.normalization_sample
        plus_norm_data = plus_norm_sample.as_dict()
        minus_norm_data = minus_norm_sample.as_dict()
        j_plus_values = jnp.asarray(background.plus_shape(plus_norm_data))
        j_minus_values = jnp.asarray(background.resolved_minus_shape(minus_norm_data))
        if background.apply_veto:
            if plus_veto is not None:
                j_plus_values = j_plus_values * jnp.asarray(plus_veto(plus_norm_data))
            if minus_veto is not None:
                j_minus_values = j_minus_values * jnp.asarray(minus_veto(minus_norm_data))
        j_plus = float(jnp.mean(jnp.asarray(plus_norm_sample.weights) * j_plus_values))
        j_minus = float(jnp.mean(jnp.asarray(minus_norm_sample.weights) * j_minus_values))
        plus_probability = j_plus / (j_plus + j_minus)
        count_plus = int(rng.binomial(int(count), plus_probability))
        count_minus = int(count) - count_plus

        if count_plus > 0:
            def plus_background_density(data, background=background):
                result = jnp.asarray(background.plus_shape(data))
                if background.apply_veto and plus_veto is not None:
                    result = result * jnp.asarray(plus_veto(data))
                return result

            plus_samples.append(
                _accept_reject_component(
                    plus_model,
                    count_plus,
                    plus_background_density,
                    seed=_derived_seed(seed, 100 + index),
                    pool_size=pool_size,
                    batch_size=batch_size,
                    envelope_safety=envelope_safety,
                    max_restarts=max_restarts,
                )
            )

        if count_minus > 0:
            def minus_background_density(data, background=background):
                result = jnp.asarray(background.resolved_minus_shape(data))
                if background.apply_veto and minus_veto is not None:
                    result = result * jnp.asarray(minus_veto(data))
                return result

            minus_samples.append(
                _accept_reject_component(
                    minus_model,
                    count_minus,
                    minus_background_density,
                    seed=_derived_seed(seed, 200 + index),
                    pool_size=pool_size,
                    batch_size=batch_size,
                    envelope_safety=envelope_safety,
                    max_restarts=max_restarts,
                )
            )

    def finish(samples: list[PhaseSpaceSample], model, key_seed: int) -> PhaseSpaceSample:
        if not samples:
            return _empty_sample(
                model,
                _derived_seed(seed, 500 + key_seed),
                include_momenta=include_momenta,
            )
        toy = _merge_samples(samples)
        if shuffle and toy.size > 1:
            toy = toy.take(
                jax.random.permutation(
                    jax.random.key(
                        (200 if seed is None else int(seed) + 200 + key_seed) % (2**32)
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
        if include_momenta:
            toy = _attach_momenta(
                model,
                toy,
                _derived_seed(seed, 910_000 + key_seed),
            )
        return toy

    plus_toy = finish(plus_samples, plus_model, 0)
    minus_toy = finish(minus_samples, minus_model, 1)
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
