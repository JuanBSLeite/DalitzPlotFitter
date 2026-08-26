"""First end-to-end DalitzPlotFitter example: D+ -> pi+ pi+ pi- via rho(770)0.

This channel is also used in the AmpForm documentation to demonstrate automatic
symmetrization of the two indistinguishable pi+ pi- subsystems.
"""

import jax
import jax.numpy as jnp

from dalitzplotfitter.amplitude import (
    AmplitudeBuilder,
    compile_model,
    create_kinematic_transformer,
)
from dalitzplotfitter.config import enable_x64
from dalitzplotfitter.kinematics import ThreeBodyPhaseSpace
from dalitzplotfitter.reaction import ReactionBuilder


def main() -> None:
    enable_x64()

    reaction = ReactionBuilder(
        initial_state="D+",
        final_state=["pi+", "pi+", "pi-"],
        allowed_intermediate_particles=["rho(770)0"],
    ).build()

    # QRules removes indistinguishable quantum-state transitions. AmpForm restores
    # the kinematically distinct pi+ pi- pairings when formulating the amplitude.
    model = AmplitudeBuilder(reaction).build()
    intensity = compile_model(model)
    transformer = create_kinematic_transformer(model)

    phase_space = ThreeBodyPhaseSpace.from_reaction(reaction)
    sample = phase_space.generate(jax.random.key(7), 100_000)
    kinematic_data = transformer(sample.as_momentum_dict())

    values = intensity(kinematic_data)
    normalization = jnp.mean(sample.weights * values)

    print("Reaction: D+ -> pi+ pi+ pi-")
    print(f"QRules transitions: {len(reaction.transitions)}")
    print(f"AmpForm amplitudes: {len(model.amplitudes)}")
    print(f"Phase-space events: {sample.size}")
    print(f"Intensity min/max: {float(jnp.min(values)):.6g} / {float(jnp.max(values)):.6g}")
    print(f"MC normalization: {float(normalization):.8g}")
    print("Parameters:")
    for name, value in intensity.parameters.items():
        print(f"  {name} = {value}")


if __name__ == "__main__":
    main()
