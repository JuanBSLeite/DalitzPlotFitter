import jax
import jax.numpy as jnp

from dalitzplotfitter.amplitude import (
    AmplitudeBuilder,
    compile_model,
    create_kinematic_transformer,
)
from dalitzplotfitter.kinematics import ThreeBodyPhaseSpace
from dalitzplotfitter.reaction import ReactionBuilder


def test_dplus_rho_model_evaluates_on_native_phase_space():
    reaction = ReactionBuilder(
        initial_state="D+",
        final_state=["pi+", "pi+", "pi-"],
        allowed_intermediate_particles=["rho(770)0"],
    ).build()
    model = AmplitudeBuilder(reaction).build()

    # QRules has one quantum-state transition, while AmpForm symmetrizes the two
    # kinematically distinct pi+ pi- pairings inside the amplitude expression.
    assert len(reaction.transitions) == 1
    assert len(model.amplitudes) == 1

    compiled = compile_model(model)
    transformer = create_kinematic_transformer(model)
    sample = ThreeBodyPhaseSpace.from_reaction(reaction).generate(
        jax.random.key(3),
        256,
    )
    data = transformer(sample.as_momentum_dict())
    values = jnp.asarray(compiled(data))

    assert values.shape == (sample.size,)
    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(values >= 0.0)
    assert jnp.mean(sample.weights * values) > 0.0
