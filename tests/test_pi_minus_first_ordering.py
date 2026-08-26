import numpy as np
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


def _build(resonance: str):
    reaction = ReactionBuilder(
        initial_state="D+",
        final_state=["pi-", "pi+", "pi+"],
        allowed_intermediate_particles=[resonance],
    ).build()
    model = AmplitudeBuilder(reaction).build()
    return reaction, model, compile_amplitude_component(model)


def test_pi_minus_first_component_dynamics_are_finite():
    enable_x64()
    rho_reaction, rho_model, rho = _build("rho(770)0")
    _, _, f0 = _build("f(0)(980)")

    sample = ThreeBodyPhaseSpace.from_reaction(rho_reaction).generate(
        jax.random.key(2026), 20_000
    )
    transformer = create_kinematic_transformer(rho_model)
    data = transformer(sample.as_momentum_dict())

    rho_values = np.asarray(rho(data, None))
    f0_values = np.asarray(f0(data, None))
    nr_values = np.asarray(ConstantAmplitude()(data, None))

    rho_finite = np.isfinite(rho_values)
    f0_finite = np.isfinite(f0_values)
    nr_finite = np.isfinite(nr_values)

    assert rho_finite.all(), (
        f"rho has {np.count_nonzero(~rho_finite)}/{rho_values.size} non-finite values"
    )
    assert f0_finite.all(), (
        f"f0 has {np.count_nonzero(~f0_finite)}/{f0_values.size} non-finite values"
    )
    assert nr_finite.all()

    amplitude = rho_values + 0.55 * np.exp(1.15j) * f0_values + 0.28 * np.exp(-0.85j) * nr_values
    intensity = np.real(amplitude * np.conj(amplitude))
    finite = np.isfinite(intensity)
    assert finite.all(), (
        f"coherent intensity has {np.count_nonzero(~finite)}/{intensity.size} non-finite values"
    )
    assert np.all(intensity >= 0.0)
    assert np.sum(np.asarray(sample.weights) * intensity) > 0.0
