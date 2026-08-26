"""D+ -> pi+ pi+ pi- with rho(770)0, f0(980), and non-resonant terms.

AmpForm supplies only the dynamical functions F_i(x). DalitzPlotFitter owns the
complex coefficients c_i and constructs the coherent sum

    A(x) = c_rho F_rho(x) + c_f0 F_f0(x) + c_NR.
"""

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


def build_resonance(resonance: str):
    reaction = ReactionBuilder(
        initial_state="D+",
        final_state=["pi+", "pi+", "pi-"],
        allowed_intermediate_particles=[resonance],
    ).build()
    model = AmplitudeBuilder(reaction).build()
    return reaction, model, compile_amplitude_component(model)


def main() -> None:
    enable_x64()

    rho_reaction, rho_model, rho_dynamics = build_resonance("rho(770)0")
    _, f0_model, f0_dynamics = build_resonance("f(0)(980)")

    # The rho coefficient defines the arbitrary global magnitude and phase reference.
    rho = AmplitudeComponent(
        "rho(770)0",
        rho_dynamics,
        MagPhase(r=1.0, phi=0.0),
    )
    f0 = AmplitudeComponent(
        "f(0)(980)",
        f0_dynamics,
        MagPhase(r=0.50, phi=2.0),
    )
    non_resonant = AmplitudeComponent(
        "NR",
        ConstantAmplitude(),
        MagPhase(r=0.30, phi=-1.0),
    )
    amplitude = CoherentAmplitudeModel((rho, f0, non_resonant))

    phase_space = ThreeBodyPhaseSpace.from_reaction(rho_reaction)
    sample = phase_space.generate(jax.random.key(11), 100_000)

    # rho requires the richest angular kinematics of the three components, so its
    # transformer contains all variables needed by f0 as well.
    transformer = create_kinematic_transformer(rho_model)
    data = transformer(sample.as_momentum_dict())

    total_intensity = amplitude.intensity(data)
    component_amplitudes = amplitude.component_amplitudes(data)
    incoherent_intensity = sum(
        jnp.abs(value) ** 2 for value in component_amplitudes.values()
    )
    interference = total_intensity - incoherent_intensity

    normalization = jnp.mean(sample.weights * total_intensity)
    incoherent_normalization = jnp.mean(sample.weights * incoherent_intensity)
    interference_integral = jnp.mean(sample.weights * interference)

    print("D+ -> pi+ pi+ pi- amplitude model")
    print("  rho(770)0 : MagPhase(r=1.00, phi=0.00) [reference]")
    print("  f(0)(980) : MagPhase(r=0.50, phi=2.00)")
    print("  NR        : MagPhase(r=0.30, phi=-1.00)")
    print(f"Phase-space events: {sample.size}")
    print(f"Coherent normalization:   {float(normalization):.8g}")
    print(f"Incoherent normalization: {float(incoherent_normalization):.8g}")
    print(f"Interference integral:     {float(interference_integral):.8g}")

    print("Dynamic parameters (external c_i are intentionally absent):")
    for label, dynamics in (("rho", rho_dynamics), ("f0", f0_dynamics)):
        print(f"  {label}:")
        for name, value in dynamics.parameters.items():
            print(f"    {name} = {value}")

    # Keep these objects referenced to make it easy to inspect them interactively.
    _ = f0_model


if __name__ == "__main__":
    main()
