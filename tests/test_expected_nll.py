"""Deterministic checks of the amplitude normalization in the expected NLL."""

import jax
import jax.numpy as jnp

from dalitzplotfitter import ThreeBodyPhaseSpace, enable_x64
from dalitzplotfitter.amplitude import (
    AmplitudeBuilder,
    ConstantAmplitude,
    compile_amplitude_component,
    create_kinematic_transformer,
)
from dalitzplotfitter.reaction import ReactionBuilder


enable_x64()


def _build_resonance(resonance: str):
    reaction = ReactionBuilder(
        initial_state="D+",
        final_state=["pi-", "pi+", "pi+"],
        allowed_intermediate_particles=[resonance],
    ).build()
    model = AmplitudeBuilder(reaction).build()
    return reaction, model, compile_amplitude_component(model)


def test_expected_nll_is_stationary_at_injected_coefficients():
    """The exact discrete expected NLL must be stationary at its truth point.

    The same weighted phase-space sample defines both the truth distribution and
    the candidate normalization. This removes toy fluctuations entirely and is
    therefore a direct regression test of the normalization convention.
    """

    rho_reaction, rho_model, rho = _build_resonance("rho(770)0")
    _, _, f0 = _build_resonance("f(0)(980)")
    phase_space = ThreeBodyPhaseSpace.from_reaction(rho_reaction)
    sample = phase_space.generate(jax.random.key(91), 40_000)
    transformer = create_kinematic_transformer(rho_model)
    data = transformer(sample.as_momentum_dict())

    components = jnp.stack(
        [
            jnp.asarray(rho(data, None)),
            jnp.asarray(f0(data, None)),
            jnp.asarray(ConstantAmplitude()(data, None)),
        ],
        axis=1,
    )
    weights = jnp.asarray(sample.weights)

    truth = jnp.asarray([0.55, 1.15, 0.28, -0.85])

    def coefficients(vector):
        return jnp.asarray(
            [
                1.0 + 0.0j,
                vector[0] * jnp.exp(1j * vector[1]),
                vector[2] * jnp.exp(1j * vector[3]),
            ]
        )

    truth_amplitude = components @ coefficients(truth)
    truth_intensity = jnp.abs(truth_amplitude) ** 2
    truth_measure = weights * truth_intensity
    truth_measure_sum = jnp.sum(truth_measure)

    def expected_nll(vector):
        amplitude = components @ coefficients(vector)
        intensity = jnp.abs(amplitude) ** 2
        normalization = jnp.mean(weights * intensity)
        cross_entropy = -jnp.sum(
            truth_measure * jnp.log(jnp.clip(intensity, min=1e-300))
        ) / truth_measure_sum
        return cross_entropy + jnp.log(normalization)

    gradient = jax.grad(expected_nll)(truth)
    assert jnp.allclose(gradient, jnp.zeros_like(gradient), rtol=0.0, atol=2e-10)

    # Displacing the f0 magnitude must increase the expected NLL on the exact
    # same discrete measure; otherwise numerator and normalization conventions
    # are inconsistent.
    lower_f0 = truth.at[0].set(0.25)
    assert expected_nll(lower_f0) > expected_nll(truth)
