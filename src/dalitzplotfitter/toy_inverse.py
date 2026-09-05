"""High-level inverse-transform pseudo-data generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.inverse_transform import DalitzInverseTransformSampler
from dalitzplotfitter.kinematics import PhaseSpaceSample
from dalitzplotfitter.toy_accept import (
    CPToyBackground,
    ToyBackground,
    _acceptance,
    _background_weights,
    _derived_seed,
    _empty_sample,
    _integral,
    _merge_samples,
)


def _prepare_sampler(
    model,
    density_function,
    *,
    resolution: int,
    quantile_resolution: int | None,
) -> DalitzInverseTransformSampler:
    return DalitzInverseTransformSampler.prepare(
        model.channel.parent_mass,
        model.channel.daughter_masses,
        density_function,
        resolution=resolution,
        quantile_resolution=quantile_resolution,
    )


def _shuffle(sample: PhaseSpaceSample, *, seed: int | None, offset: int) -> PhaseSpaceSample:
    if sample.size <= 1:
        return sample
    key_seed = offset if seed is None else (int(seed) + offset) % (2**32)
    selected = sample.take(jax.random.permutation(jax.random.key(key_seed), sample.size))
    return PhaseSpaceSample(
        s12=selected.s12,
        s13=selected.s13,
        s23=selected.s23,
        weights=jnp.ones((selected.size,), dtype=selected.s12.dtype),
        p1=selected.p1,
        p2=selected.p2,
        p3=selected.p3,
    )


@dataclass(frozen=True)
class PreparedInverseToyGenerator:
    """Reusable non-CP inverse-transform generator for fixed model parameters.

    Preparing the generator evaluates the requested signal and background
    densities on the Dalitz CDF grids once. Repeated calls to ``generate`` then
    only draw uniform random numbers, invert the tabulated CDFs, reconstruct
    four-momenta, and mix the already prepared components.
    """

    signal_sampler: DalitzInverseTransformSampler
    background_samplers: tuple[DalitzInverseTransformSampler, ...] = ()
    backgrounds: tuple[ToyBackground, ...] = ()
    signal_fraction: float = 1.0

    def generate(
        self,
        size: int,
        *,
        seed: int | None = None,
        shuffle: bool = True,
        include_momenta: bool = True,
    ) -> PhaseSpaceSample:
        if size <= 0:
            raise ValueError("toy size must be positive")
        rng = np.random.default_rng(seed)
        n_signal = (
            int(rng.binomial(size, float(self.signal_fraction)))
            if self.backgrounds
            else size
        )
        n_background = size - n_signal
        background_weights = _background_weights(self.backgrounds)
        counts = (
            rng.multinomial(n_background, background_weights)
            if n_background > 0
            else np.zeros(len(self.backgrounds), dtype=int)
        )
        samples: list[PhaseSpaceSample] = []
        if n_signal:
            samples.append(
                self.signal_sampler.generate(
                    n_signal,
                    seed=_derived_seed(seed, 1),
                    include_momenta=include_momenta,
                )
            )
        for index, (sampler, count) in enumerate(zip(self.background_samplers, counts)):
            if int(count):
                samples.append(
                    sampler.generate(
                        int(count),
                        seed=_derived_seed(seed, 100 + index),
                        include_momenta=include_momenta,
                    )
                )
        toy = _merge_samples(samples)
        return _shuffle(toy, seed=seed, offset=100) if shuffle else toy


def prepare_inverse_toy_generator(
    model,
    *,
    parameters: Mapping[str, object] | None = None,
    efficiency=None,
    veto=None,
    signal_fraction: float = 1.0,
    backgrounds: Sequence[ToyBackground] = (),
    resolution: int = 1024,
    quantile_resolution: int | None = None,
) -> PreparedInverseToyGenerator:
    """Prepare reusable inverse CDFs for a fixed non-CP toy model."""

    if not 0.0 <= float(signal_fraction) <= 1.0:
        raise ValueError("signal_fraction must lie in [0, 1]")
    backgrounds = tuple(backgrounds)
    if float(signal_fraction) < 1.0 and not backgrounds:
        raise ValueError("signal_fraction < 1 requires at least one background")
    if backgrounds and float(signal_fraction) >= 1.0:
        raise ValueError("backgrounds require signal_fraction < 1")
    values = {} if parameters is None else parameters

    def signal_density(data):
        return _acceptance(efficiency, veto, data) * jnp.asarray(
            model.intensity(data, values)
        )

    signal_sampler = _prepare_sampler(
        model,
        signal_density,
        resolution=resolution,
        quantile_resolution=quantile_resolution,
    )
    prepared_backgrounds = []
    for background in backgrounds:
        def background_density(data, background=background):
            result = jnp.asarray(background.shape(data))
            if veto is not None and background.apply_veto:
                result = result * jnp.asarray(veto(data))
            return result

        prepared_backgrounds.append(
            _prepare_sampler(
                model,
                background_density,
                resolution=resolution,
                quantile_resolution=quantile_resolution,
            )
        )
    return PreparedInverseToyGenerator(
        signal_sampler=signal_sampler,
        background_samplers=tuple(prepared_backgrounds),
        backgrounds=backgrounds,
        signal_fraction=float(signal_fraction),
    )


def generate_signal_toy_inverse(
    model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    efficiency=None,
    veto=None,
    seed: int | None = None,
    resolution: int = 1024,
    quantile_resolution: int | None = None,
    include_momenta: bool = True,
) -> PhaseSpaceSample:
    prepared = prepare_inverse_toy_generator(
        model,
        parameters=parameters,
        efficiency=efficiency,
        veto=veto,
        resolution=resolution,
        quantile_resolution=quantile_resolution,
    )
    return prepared.generate(
        size,
        seed=seed,
        shuffle=False,
        include_momenta=include_momenta,
    )


def generate_toy_inverse(
    model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    efficiency=None,
    veto=None,
    signal_fraction: float = 1.0,
    backgrounds: Sequence[ToyBackground] = (),
    seed: int | None = None,
    shuffle: bool = True,
    resolution: int = 1024,
    quantile_resolution: int | None = None,
    include_momenta: bool = True,
) -> PhaseSpaceSample:
    prepared = prepare_inverse_toy_generator(
        model,
        parameters=parameters,
        efficiency=efficiency,
        veto=veto,
        signal_fraction=signal_fraction,
        backgrounds=backgrounds,
        resolution=resolution,
        quantile_resolution=quantile_resolution,
    )
    return prepared.generate(
        size,
        seed=seed,
        shuffle=shuffle,
        include_momenta=include_momenta,
    )


def generate_cp_toy_inverse(
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
    shuffle: bool = True,
    resolution: int = 1024,
    quantile_resolution: int | None = None,
    include_momenta: bool = True,
) -> tuple[PhaseSpaceSample, PhaseSpaceSample]:
    """Generate a simultaneous CP toy using inverse-transform component samplers."""

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

    def plus_signal_density(data):
        return _acceptance(plus_efficiency, plus_veto, data) * jnp.asarray(
            plus_model.intensity(data, values)
        )

    def minus_signal_density(data):
        return _acceptance(minus_efficiency, minus_veto, data) * jnp.asarray(
            minus_model.intensity(data, values)
        )

    plus_signal_sampler = _prepare_sampler(
        plus_model,
        plus_signal_density,
        resolution=resolution,
        quantile_resolution=quantile_resolution,
    )
    minus_signal_sampler = _prepare_sampler(
        minus_model,
        minus_signal_density,
        resolution=resolution,
        quantile_resolution=quantile_resolution,
    )

    plus_samples: list[PhaseSpaceSample] = []
    minus_samples: list[PhaseSpaceSample] = []
    if n_signal_plus:
        plus_samples.append(
            plus_signal_sampler.generate(
                n_signal_plus,
                seed=_derived_seed(seed, 10),
                include_momenta=include_momenta,
            )
        )
    if n_signal_minus:
        minus_samples.append(
            minus_signal_sampler.generate(
                n_signal_minus,
                seed=_derived_seed(seed, 20),
                include_momenta=include_momenta,
            )
        )

    bg_mix = _background_weights(backgrounds)
    bg_counts = (
        rng.multinomial(n_background, bg_mix)
        if n_background > 0
        else np.zeros(len(backgrounds), dtype=int)
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

        if count_plus:
            def plus_background_density(data, background=background):
                result = jnp.asarray(background.plus_shape(data))
                if background.apply_veto and plus_veto is not None:
                    result = result * jnp.asarray(plus_veto(data))
                return result

            sampler = _prepare_sampler(
                plus_model,
                plus_background_density,
                resolution=resolution,
                quantile_resolution=quantile_resolution,
            )
            plus_samples.append(
                sampler.generate(
                    count_plus,
                    seed=_derived_seed(seed, 100 + index),
                    include_momenta=include_momenta,
                )
            )
        if count_minus:
            def minus_background_density(data, background=background):
                result = jnp.asarray(background.resolved_minus_shape(data))
                if background.apply_veto and minus_veto is not None:
                    result = result * jnp.asarray(minus_veto(data))
                return result

            sampler = _prepare_sampler(
                minus_model,
                minus_background_density,
                resolution=resolution,
                quantile_resolution=quantile_resolution,
            )
            minus_samples.append(
                sampler.generate(
                    count_minus,
                    seed=_derived_seed(seed, 200 + index),
                    include_momenta=include_momenta,
                )
            )

    def finish(samples, model, offset):
        if not samples:
            return _empty_sample(model, _derived_seed(seed, 500 + offset))
        result = _merge_samples(samples)
        return _shuffle(result, seed=seed, offset=200 + offset) if shuffle else result

    plus_toy = finish(plus_samples, plus_model, 0)
    minus_toy = finish(minus_samples, minus_model, 1)
    if plus_toy.size + minus_toy.size != size:
        raise RuntimeError("internal inverse-transform CP toy count mismatch")
    return plus_toy, minus_toy


__all__ = [
    "PreparedInverseToyGenerator",
    "generate_cp_toy_inverse",
    "generate_signal_toy_inverse",
    "generate_toy_inverse",
    "prepare_inverse_toy_generator",
]
