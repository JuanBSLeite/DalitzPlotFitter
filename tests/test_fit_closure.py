"""End-to-end toy-MC closure test for the D+ reference amplitude model."""

import numpy as np
import jax
import jax.numpy as jnp

from dalitzplotfitter import ThreeBodyPhaseSpace, enable_x64
from dalitzplotfitter.amplitude import (
    AmplitudeBuilder,
    AmplitudeComponent,
    ConstantAmplitude,
    PreparedAmplitudeCache,
    compile_amplitude_component,
    create_kinematic_transformer,
)
from dalitzplotfitter.coefficients import FitMagPhase
from dalitzplotfitter.fit import Minimizer, Parameter
from dalitzplotfitter.reaction import ReactionBuilder
from dalitzplotfitter.toy import ToyGenerator


def _build_resonance(resonance: str):
    reaction = ReactionBuilder(
        initial_state="D+",
        final_state=["pi-", "pi+", "pi+"],
        allowed_intermediate_particles=[resonance],
    ).build()
    model = AmplitudeBuilder(reaction).build()
    return reaction, model, compile_amplitude_component(model)


def _wrapped_delta(phi_fit: float, phi_true: float) -> float:
    return float(np.angle(np.exp(1j * (phi_fit - phi_true))))


def test_dplus_toy_fit_recovers_injected_mag_phase_parameters():
    """Generate -> randomize start -> fit -> compare to injected truth."""

    enable_x64()

    rho_reaction, rho_model, rho_dynamics = _build_resonance("rho(770)0")
    _, _, f0_dynamics = _build_resonance("f(0)(980)")

    # Reference amplitude: fixes the arbitrary overall phase and scale.
    rho_r = Parameter.coefficient("rho.r", 1.0, fixed=True, owner="rho")
    rho_phi = Parameter.coefficient("rho.phi", 0.0, fixed=True, owner="rho")

    truth = {
        "f0.r": 0.55,
        "f0.phi": 1.15,
        "nr.r": 0.28,
        "nr.phi": -0.85,
    }

    # Randomized, reproducible starting point intentionally displaced from truth.
    rng = np.random.default_rng(314159)
    f0_r = Parameter.coefficient(
        "f0.r",
        float(rng.uniform(0.25, 0.90)),
        bounds=(0.0, 1.5),
        step=0.02,
        owner="f0",
    )
    f0_phi = Parameter.coefficient(
        "f0.phi",
        float(rng.uniform(-2.8, 2.8)),
        bounds=(-np.pi, np.pi),
        step=0.05,
        owner="f0",
    )
    nr_r = Parameter.coefficient(
        "nr.r",
        float(rng.uniform(0.08, 0.65)),
        bounds=(0.0, 1.0),
        step=0.02,
        owner="NR",
    )
    nr_phi = Parameter.coefficient(
        "nr.phi",
        float(rng.uniform(-2.8, 2.8)),
        bounds=(-np.pi, np.pi),
        step=0.05,
        owner="NR",
    )

    rho_coefficient = FitMagPhase(rho_r, rho_phi)
    f0_coefficient = FitMagPhase(f0_r, f0_phi)
    nr_coefficient = FitMagPhase(nr_r, nr_phi)

    components = (
        AmplitudeComponent("rho", rho_dynamics, rho_coefficient),
        AmplitudeComponent("f0", f0_dynamics, f0_coefficient),
        AmplitudeComponent("NR", ConstantAmplitude(), nr_coefficient),
    )

    phase_space = ThreeBodyPhaseSpace.from_reaction(rho_reaction)
    transformer = create_kinematic_transformer(rho_model)

    # Generation is deliberately independent of PreparedAmplitudeCache.
    def toy_intensity(data, values):
        rho = rho_dynamics(data, None)
        f0 = f0_dynamics(data, None)
        nr = ConstantAmplitude()(data, None)
        amplitude = (
            rho_coefficient.value(values=values) * rho
            + f0_coefficient.value(values=values) * f0
            + nr_coefficient.value(values=values) * nr
        )
        return jnp.real(amplitude * jnp.conj(amplitude))

    generator = ToyGenerator(
        phase_space=phase_space,
        transformer=transformer,
        pool_size=60_000,
    )
    toy_sample, toy_data = generator.generate(
        jax.random.key(2026),
        size=3_000,
        intensity=toy_intensity,
        parameters=truth,
    )

    # Independent normalization sample: never reuse the generation pool.
    normalization_sample = phase_space.generate(jax.random.key(2027), 60_000)
    normalization_data = transformer(normalization_sample.as_momentum_dict())

    parameters = (rho_r, rho_phi, f0_r, f0_phi, nr_r, nr_phi)
    cache = PreparedAmplitudeCache.prepare(
        components,
        data=toy_data,
        normalization_data=normalization_data,
        normalization_weights=normalization_sample.weights,
        parameters=parameters,
    )

    def nll(values):
        intensity, normalization = cache.evaluate(values)
        return -jnp.sum(jnp.log(jnp.clip(intensity, min=1e-300))) + (
            toy_sample.size * jnp.log(normalization)
        )

    result = Minimizer(nll, parameters).fit()

    assert result.valid
    assert abs(result.values["f0.r"] - truth["f0.r"]) < 0.12
    assert abs(_wrapped_delta(result.values["f0.phi"], truth["f0.phi"])) < 0.22
    assert abs(result.values["nr.r"] - truth["nr.r"]) < 0.10
    assert abs(_wrapped_delta(result.values["nr.phi"], truth["nr.phi"])) < 0.28
