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
from dalitzplotfitter.coefficients import RealImag
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


def _assert_one_sigma_compatibility(result, name: str, generated: float):
    fitted = float(result.values[name])
    error = float(result.errors[name])
    assert np.isfinite(error) and error > 0.0, f"invalid HESSE error for {name}: {error}"
    pull = (generated - fitted) / error
    assert abs(pull) < 1.0, (
        f"{name} is not compatible with the generated value within 1 sigma: "
        f"generated={generated}, fitted={fitted}, error={error}, pull={pull}"
    )


def test_dplus_toy_fit_recovers_injected_real_imag_parameters():
    """Generate -> multi-start RealImag fit -> require one-sigma closure."""

    enable_x64()

    rho_reaction, rho_model, rho_dynamics = _build_resonance("rho(770)0")
    _, _, f0_dynamics = _build_resonance("f(0)(980)")

    # Same physical truth as the former polar benchmark, expressed as x + i y.
    truth = {
        "f0.x": 0.55 * np.cos(1.15),
        "f0.y": 0.55 * np.sin(1.15),
        "nr.x": 0.28 * np.cos(-0.85),
        "nr.y": 0.28 * np.sin(-0.85),
    }

    rho_x = Parameter.coefficient("rho.x", 1.0, fixed=True, owner="rho")
    rho_y = Parameter.coefficient("rho.y", 0.0, fixed=True, owner="rho")

    rng = np.random.default_rng(314159)
    f0_x = Parameter.coefficient("f0.x", float(rng.uniform(-0.8, 0.8)), bounds=(-1.5, 1.5), step=0.02, owner="f0")
    f0_y = Parameter.coefficient("f0.y", float(rng.uniform(-0.8, 0.8)), bounds=(-1.5, 1.5), step=0.02, owner="f0")
    nr_x = Parameter.coefficient("nr.x", float(rng.uniform(-0.6, 0.6)), bounds=(-1.0, 1.0), step=0.02, owner="NR")
    nr_y = Parameter.coefficient("nr.y", float(rng.uniform(-0.6, 0.6)), bounds=(-1.0, 1.0), step=0.02, owner="NR")

    rho_coefficient = RealImag(rho_x, rho_y)
    f0_coefficient = RealImag(f0_x, f0_y)
    nr_coefficient = RealImag(nr_x, nr_y)

    components = (
        AmplitudeComponent("rho", rho_dynamics, rho_coefficient),
        AmplitudeComponent("f0", f0_dynamics, f0_coefficient),
        AmplitudeComponent("NR", ConstantAmplitude(), nr_coefficient),
    )

    phase_space = ThreeBodyPhaseSpace.from_reaction(rho_reaction)
    transformer = create_kinematic_transformer(rho_model)

    def toy_intensity(data, values):
        amplitude = (
            rho_coefficient.value(values=values) * rho_dynamics(data, None)
            + f0_coefficient.value(values=values) * f0_dynamics(data, None)
            + nr_coefficient.value(values=values) * ConstantAmplitude()(data, None)
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

    parameters = (rho_x, rho_y, f0_x, f0_y, nr_x, nr_y)
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
    assert jnp.allclose(
        cache.normalization(truth),
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
        {"f0.x": 0.2, "f0.y": 0.5, "nr.x": 0.2, "nr.y": -0.2},
        {"f0.x": -0.5, "f0.y": 0.5, "nr.x": 0.3, "nr.y": 0.3},
        {"f0.x": 0.5, "f0.y": -0.5, "nr.x": -0.3, "nr.y": -0.3},
    )
    results = [minimizer.fit(start_values=start) for start in starts]
    valid_results = [result for result in results if result.valid]
    assert valid_results, "No valid Minuit minimum was found"
    result = min(valid_results, key=lambda candidate: float(candidate.fval))

    nll_truth = float(nll(truth))
    delta_nll_truth = nll_truth - float(result.fval)
    assert delta_nll_truth < 25.0, (
        f"Injected truth is far above the best minimum: "
        f"NLL(truth)={nll_truth}, NLL(best)={float(result.fval)}, "
        f"DeltaNLL={delta_nll_truth}"
    )

    for name in ("f0.x", "f0.y", "nr.x", "nr.y"):
        _assert_one_sigma_compatibility(result, name, truth[name])
