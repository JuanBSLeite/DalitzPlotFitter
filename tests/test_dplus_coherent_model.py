import jax
import jax.numpy as jnp

from dalitzplotfitter import MagPhase, ThreeBodyPhaseSpace, enable_x64
from dalitzplotfitter.amplitude import (
    AmplitudeBuilder,
    AmplitudeComponent,
    CoherentAmplitudeModel,
    ConstantAmplitude,
    compile_amplitude_component,
    create_kinematic_transformer,
)
from dalitzplotfitter.reaction import ReactionBuilder


def _build_component(resonance: str):
    reaction = ReactionBuilder(
        initial_state="D+",
        final_state=["pi+", "pi+", "pi-"],
        allowed_intermediate_particles=[resonance],
    ).build()
    model = AmplitudeBuilder(reaction).build()
    return reaction, model, compile_amplitude_component(model)


def test_dplus_rho_f0_nr_is_coherent_and_finite():
    enable_x64()

    rho_reaction, rho_model, rho_dynamics = _build_component("rho(770)0")
    _, _, f0_dynamics = _build_component("f(0)(980)")

    model = CoherentAmplitudeModel(
        (
            AmplitudeComponent(
                "rho(770)0", rho_dynamics, MagPhase(r=1.0, phi=0.0)
            ),
            AmplitudeComponent(
                "f(0)(980)", f0_dynamics, MagPhase(r=0.5, phi=2.0)
            ),
            AmplitudeComponent(
                "NR", ConstantAmplitude(), MagPhase(r=0.3, phi=-1.0)
            ),
        )
    )

    sample = ThreeBodyPhaseSpace.from_reaction(rho_reaction).generate(
        jax.random.key(19),
        128,
    )
    transformer = create_kinematic_transformer(rho_model)
    data = transformer(sample.as_momentum_dict())

    total = model.intensity(data)
    parts = model.component_amplitudes(data)
    incoherent = sum(jnp.abs(value) ** 2 for value in parts.values())
    interference = total - incoherent

    assert total.shape == (sample.size,)
    assert jnp.all(jnp.isfinite(total))
    assert jnp.all(total >= 0.0)
    assert jnp.mean(sample.weights * total) > 0.0
    assert jnp.max(jnp.abs(interference)) > 1e-10
