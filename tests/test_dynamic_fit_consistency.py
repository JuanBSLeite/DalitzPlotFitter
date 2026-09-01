import math

import jax
import jax.numpy as jnp
import numpy as np

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    Minimizer,
    NonResonant,
    Parameter,
    RealImag,
    Resonance,
    SquareDalitzGrid,
    enable_x64,
)


enable_x64()


def _dynamic_rho_model(*, coefficient_free: bool = True):
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    mass = Parameter.dynamics(
        "rho.mass",
        0.760,
        owner="rho",
        bounds=(0.72, 0.82),
        step=0.001,
    )
    width = Parameter.dynamics(
        "rho.width",
        0.180,
        owner="rho",
        bounds=(0.08, 0.25),
        step=0.001,
    )
    if coefficient_free:
        coefficient = RealImag(
            Parameter.coefficient(
                "rho.x", 0.65, owner="rho", bounds=(-1.5, 1.5), step=0.01
            ),
            Parameter.coefficient(
                "rho.y", -0.25, owner="rho", bounds=(-1.5, 1.5), step=0.01
            ),
        )
    else:
        coefficient = RealImag(0.65, -0.25)

    return DecayModel(
        channel,
        [
            Resonance(
                "rho",
                pair=(0, 1),
                coefficient=coefficient,
                mass=mass,
                width=width,
                spin=1,
                resonance_radius=3.0,
                parent_radius=0.0,
            ),
            NonResonant(RealImag(1.0, 0.0)),
        ],
        normalization_resolution=64,
    )


def _grid(model, resolution):
    return SquareDalitzGrid(
        model.channel.parent_mass,
        model.channel.daughter_masses,
        resolution=resolution,
    ).sample()


def _nll_from_cache(cache, n_data):
    def nll(values):
        intensity, normalization = cache.evaluate(values)
        return -jnp.sum(jnp.log(intensity)) + n_data * jnp.log(normalization)

    return nll


def test_dynamic_cache_gradient_matches_finite_difference():
    """JAX derivatives of pole mass/width agree with direct finite differences."""

    model = _dynamic_rho_model(coefficient_free=True)
    data = model.generate_phase_space(192, seed=101)
    norm = _grid(model, 48)
    cache = model.prepare_cache(data, norm)
    nll = _nll_from_cache(cache, data.size)

    point = {
        "rho.mass": 0.781,
        "rho.width": 0.137,
        "rho.x": 0.73,
        "rho.y": -0.31,
    }

    for name, step in (("rho.mass", 2e-6), ("rho.width", 2e-6)):
        def one_dimensional(value):
            values = dict(point)
            values[name] = value
            return nll(values)

        automatic = float(jax.grad(one_dimensional)(point[name]))
        plus = float(one_dimensional(point[name] + step))
        minus = float(one_dimensional(point[name] - step))
        finite_difference = (plus - minus) / (2.0 * step)
        assert math.isclose(
            automatic,
            finite_difference,
            rel_tol=2e-5,
            abs_tol=2e-5,
        )


def test_asimov_dynamic_truth_is_stationary_with_cache():
    """With identical deterministic support, truth must be stationary."""

    model = _dynamic_rho_model(coefficient_free=True)
    sample = _grid(model, 64)
    cache = model.prepare_cache(sample, sample)
    truth = {
        "rho.mass": 0.775,
        "rho.width": 0.149,
        "rho.x": 0.72,
        "rho.y": -0.28,
    }

    truth_intensity, _ = cache.evaluate(truth)
    probability = sample.weights * truth_intensity
    probability = probability / jnp.sum(probability)

    names = tuple(parameter.name for parameter in model.parameters)
    truth_vector = jnp.asarray([truth[name] for name in names])

    def asimov_objective(vector):
        values = {name: vector[i] for i, name in enumerate(names)}
        intensity, normalization = cache.evaluate(values)
        return -jnp.sum(probability * jnp.log(intensity)) + jnp.log(normalization)

    gradient = jax.grad(asimov_objective)(truth_vector)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.max(jnp.abs(gradient)) < 2e-10


def test_asimov_mass_width_fit_recovers_truth_from_displaced_starts():
    """Mass and width must be recoverable when the resonance is identifiable."""

    model = _dynamic_rho_model(coefficient_free=False)
    sample = _grid(model, 90)
    cache = model.prepare_cache(sample, sample)
    truth = {"rho.mass": 0.775, "rho.width": 0.149}

    truth_intensity, _ = cache.evaluate(truth)
    probability = sample.weights * truth_intensity
    probability = probability / jnp.sum(probability)

    def asimov_nll(values):
        intensity, normalization = cache.evaluate(values)
        return -jnp.sum(probability * jnp.log(intensity)) + jnp.log(normalization)

    result = Minimizer(
        asimov_nll,
        model.parameters,
        tolerance=1e-10,
    ).fit_multistart(
        n_starts=6,
        seed=90210,
        simplex=True,
    ).best

    assert result.valid
    assert abs(float(result.values["rho.mass"]) - truth["rho.mass"]) < 2e-6
    assert abs(float(result.values["rho.width"]) - truth["rho.width"]) < 2e-6
    assert float(result.fmin.edm) < 1e-10
