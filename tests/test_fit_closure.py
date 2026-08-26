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


FIT_SAMPLE_SIZE = 100_000
NORMALIZATION_SAMPLE_SIZE = 1_000_000


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


def _assert_closure(
    result,
    name: str,
    truth: float,
    *,
    max_pull: float,
    max_abs_delta: float,
    wrapped: bool = False,
):
    fitted = float(result.values[name])
    error = float(result.errors[name])
    assert np.isfinite(error) and error > 0.0, f"invalid HESSE error for {name}: {error}"
    delta = _wrapped_delta(fitted, truth) if wrapped else fitted - truth
    pull = delta / error
    assert abs(delta) < max_abs_delta, (
        f"{name} failed absolute closure sanity check: fit={fitted}, truth={truth}, "
        f"delta={delta}, error={error}, pull={pull}"
    )
    assert abs(pull) < max_pull, (
        f"{name} failed pull closure: fit={fitted}, truth={truth}, delta={delta}, "
        f"error={error}, pull={pull}"
    )


def test_dplus_toy_fit_recovers_injected_mag_phase_parameters():
    """Generate -> multi-start fit -> compare best minimum to injected truth."""

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
        envelope_safety=1.2,
    )
    toy_sample, toy_data = generator.generate(
        jax.random.key(2026),
        size=FIT_SAMPLE_SIZE,
        intensity=toy_intensity,
        parameters=truth,
    )

    normalization_sample = phase_space.generate(
        jax.random.key(2027),
        NORMALIZATION_SAMPLE_SIZE,
    )
    normalization_data = transformer(normalization_sample.as_momentum_dict())

    parameters = (rho_r, rho_phi, f0_r, f0_phi, nr_r, nr_phi)
    cache = PreparedAmplitudeCache.prepare(
        components,
        data=toy_data,
        normalization_data=normalization_data,
        normalization_weights=normalization_sample.weights,
        parameters=parameters,
    )

    truth_coefficients = cache.coefficient_vector(truth)
    direct_amplitude = cache.normalization_components @ truth_coefficients
    direct_normalization = jnp.mean(
        normalization_sample.weights * jnp.abs(direct_amplitude) ** 2
    )
    matrix_normalization = cache.normalization(truth)
    assert jnp.allclose(
        matrix_normalization,
        direct_normalization,
        rtol=1e-11,
        atol=1e-12,
    )

    def nll(values):
        intensity, normalization = cache.evaluate(values)
        return -jnp.sum(jnp.log(jnp.clip(intensity, min=1e-300))) + (
            toy_sample.size * jnp.log(normalization)
        )

    minimizer = Minimizer(nll, parameters)
    default_start = {parameter.name: parameter.value for parameter in parameters}
    starts = (
        default_start,
        truth,
        {"f0.r": 0.50, "f0.phi": 1.0, "nr.r": 0.30, "nr.phi": -0.7},
        {"f0.r": 0.65, "f0.phi": 2.6, "nr.r": 0.20, "nr.phi": 2.2},
        {"f0.r": 0.40, "f0.phi": -2.6, "nr.r": 0.40, "nr.phi": -2.2},
    )
    results = [minimizer.fit(start_values=start) for start in starts]
    valid_results = [result for result in results if result.valid]
    assert valid_results, "No valid Minuit minimum was found"
    result = min(valid_results, key=lambda candidate: float(candidate.fval))

    nll_truth = float(nll(truth))
    delta_nll_truth = nll_truth - float(result.fval)
    # A correctly generated finite toy can prefer a nearby point over the truth,
    # but the injected point must not be catastrophically disfavoured.
    assert delta_nll_truth < 25.0, (
        f"Injected truth is far above the best minimum: "
        f"NLL(truth)={nll_truth}, NLL(best)={float(result.fval)}, "
        f"DeltaNLL={delta_nll_truth}"
    )

    _assert_closure(
        result,
        "f0.r",
        truth["f0.r"],
        max_pull=3.5,
        max_abs_delta=0.30,
    )
    _assert_closure(
        result,
        "f0.phi",
        truth["f0.phi"],
        max_pull=3.5,
        max_abs_delta=0.60,
        wrapped=True,
    )
    _assert_closure(
        result,
        "nr.r",
        truth["nr.r"],
        max_pull=3.5,
        max_abs_delta=0.20,
    )
    _assert_closure(
        result,
        "nr.phi",
        truth["nr.phi"],
        max_pull=3.5,
        max_abs_delta=0.60,
        wrapped=True,
    )
