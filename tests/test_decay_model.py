import jax.numpy as jnp
import pytest

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    NonResonant,
    Parameter,
    RealImag,
    Resonance,
    enable_x64,
)

enable_x64()


class ConstantLineshape:
    """Minimal test plugin proving DecayModel is lineshape-agnostic."""

    def __call__(self, mass, context):
        del context
        return jnp.ones_like(mass, dtype=jnp.complex128)


def _model(channel, components, **kwargs):
    return DecayModel(
        channel,
        components,
        normalization_method="square-dalitz",
        normalization_resolution=45,
        **kwargs,
    )


def _floating_rho_model():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    mass = Parameter.dynamics(
        "rho.mass", 0.760, owner="rho", backend_name="mass", bounds=(0.72, 0.82)
    )
    width = Parameter.dynamics(
        "rho.width", 0.180, owner="rho", backend_name="width", bounds=(0.08, 0.25)
    )
    x = Parameter.coefficient("rho.x", 0.8, owner="rho")
    y = Parameter.coefficient("rho.y", 0.2, owner="rho")
    return _model(
        channel,
        [
            Resonance(
                "rho",
                pair=(0, 1),
                coefficient=RealImag(x, y),
                mass=mass,
                width=width,
                spin=1,
            ),
            NonResonant(RealImag(1.0, 0.0)),
        ],
    )


def test_decay_channel_resolves_particle_masses_in_gev():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    assert 1.86 < channel.parent_mass < 1.88
    assert all(0.139 < mass < 0.141 for mass in channel.daughter_masses)
    assert channel.final_state_ids[1] == channel.final_state_ids[2]
    assert channel.final_state_ids[0] != channel.final_state_ids[1]


def test_unphysical_decay_channel_is_rejected():
    with pytest.raises(ValueError, match="parent mass must exceed"):
        DecayChannel("pi0", ("pi0", "pi0", "pi0"))


def test_decay_model_defaults_to_gauss_legendre_component_normalization():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = DecayModel(channel, [NonResonant(RealImag(1.0, 0.0))])
    assert model.normalize_components is True
    assert model.normalization_method == "gauss-legendre"
    assert model.normalization_resolution == 1000
    assert model._normalization_sample is None


def test_internal_normalization_grid_is_lazy_and_reused():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(channel, [NonResonant(RealImag(1.0, 0.0))])
    assert model._normalization_sample is None
    first = model.normalization_sample
    second = model.normalization_sample
    assert first is second
    assert first.size == 45**2
    assert bool(jnp.any(first.weights != first.weights[0]))


def test_decay_model_rejects_unknown_normalization_methods():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    components = [NonResonant(RealImag(1.0, 0.0))]
    with pytest.raises(ValueError, match="gauss-legendre.*square-dalitz.*auto"):
        DecayModel(channel, components, normalization_method="equal_area")


def test_decay_model_rejects_adaptive_as_public_method_name():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    components = [NonResonant(RealImag(1.0, 0.0))]
    with pytest.raises(ValueError, match="gauss-legendre.*square-dalitz.*auto"):
        DecayModel(channel, components, normalization_method="adaptive")


def test_component_normalization_is_unit_diagonal_by_default():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [
            Resonance(
                "rho_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=0.775,
                width=0.149,
                spin=1,
            ),
            NonResonant(RealImag(0.3, -0.2)),
        ],
    )
    data = model.generate_phase_space(64, seed=11)
    cache = model.prepare_cache(data)
    diagonal = jnp.real(jnp.diag(cache.normalization_matrix_fixed))
    assert jnp.allclose(diagonal, jnp.ones_like(diagonal), rtol=1e-11, atol=1e-11)


def test_component_normalization_can_be_disabled():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [NonResonant(RealImag(1.0, 0.0))],
        normalize_components=False,
    )
    data = model.generate_phase_space(32, seed=12)
    cache = model.prepare_cache(data)
    assert cache.normalize_components is False


def test_component_normalization_override_reaches_amplitude_model():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [
            Resonance(
                "rho_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=0.775,
                width=0.149,
                spin=1,
                normalize_component=False,
            ),
            NonResonant(RealImag(0.3, -0.2)),
        ],
        normalize_components=True,
    )

    built = model.amplitude_model.components
    assert built[0].normalize_component is False
    assert built[1].normalize_component is None
    assert model._component_scale(built[0]) == 1.0
    assert not jnp.isclose(model._component_scale(built[1]), 1.0)


def test_decay_model_computes_and_prints_fit_fractions(capsys):
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [
            NonResonant(RealImag(1.0, 0.0), name="first"),
            NonResonant(RealImag(1.0, 0.0), name="second"),
        ],
    )

    fractions = model.fit_fractions()
    interference = model.interference_fractions()
    printed = model.print_fit_fractions(include_interference=True, precision=2)
    output = capsys.readouterr().out

    assert jnp.allclose(fractions, jnp.asarray([0.25, 0.25]))
    assert jnp.allclose(
        jnp.sum(fractions) + jnp.sum(jnp.triu(interference, k=1)),
        1.0,
    )
    assert printed == {"first": 0.25, "second": 0.25}
    assert "Fit fractions (physical)" in output
    assert "first x second" in output
    assert "50.00" in output


def test_decay_model_acceptance_weighted_fit_fractions_validate_efficiency():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [NonResonant(RealImag(1.0, 0.0), name="NR")],
    )

    fractions = model.fit_fractions(
        efficiency=lambda data: 0.5 + 0.0 * data["s12"]
    )
    assert jnp.allclose(fractions, jnp.ones((1,)))

    with pytest.raises(ValueError, match="one value per normalization point"):
        model.fit_fractions(efficiency=jnp.ones((3,)))


def test_decay_model_rejects_duplicate_component_names():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    with pytest.raises(ValueError, match="component names must be unique"):
        _model(
            channel,
            [
                NonResonant(RealImag(1.0, 0.0), name="same"),
                NonResonant(RealImag(0.2, 0.1), name="same"),
            ],
        )


def test_decay_model_rejects_ambiguous_dynamics_backend_names():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    mass = Parameter.dynamics(
        "rho.mass",
        0.775,
        owner="rho",
        backend_name="same_backend",
        bounds=(0.70, 0.85),
    )
    width = Parameter.dynamics(
        "rho.width",
        0.149,
        owner="rho",
        backend_name="same_backend",
        bounds=(0.05, 0.30),
    )
    with pytest.raises(ValueError, match="map to the same backend key"):
        _model(
            channel,
            [
                Resonance(
                    "rho",
                    pair=(0, 1),
                    coefficient=RealImag(1.0, 0.0),
                    mass=mass,
                    width=width,
                    spin=1,
                )
            ],
        )


def test_decay_model_rejects_unphysical_core_parameter_bounds():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    mass = Parameter.dynamics(
        "rho.mass",
        0.775,
        owner="rho",
        bounds=(-0.1, 1.0),
    )
    with pytest.raises(ValueError, match="lower bound for rho.mass must be positive"):
        Resonance(
            "rho",
            pair=(0, 1),
            coefficient=RealImag(1.0, 0.0),
            mass=mass,
            width=0.149,
            spin=1,
        )


def test_decay_model_builds_symmetrized_resonance_without_manual_particle_masses():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [
            Resonance(
                "rho_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=0.77526,
                width=0.1491,
                spin=1,
            ),
            NonResonant(RealImag(0.2, -0.1)),
        ],
    )
    sample = model.generate_phase_space(128, seed=17)
    values = model.intensity(sample.as_dict())
    assert values.shape == (128,)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(values >= 0.0))


def test_decay_model_accepts_custom_lineshape_plugin():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [
            Resonance(
                "rho_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                lineshape=ConstantLineshape(),
                mass=0.77526,
                width=0.1491,
                spin=1,
            )
        ],
    )
    sample = model.generate_phase_space(64, seed=19)
    values = model.intensity(sample.as_dict())
    assert values.shape == (64,)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(values >= 0.0))


def test_decay_model_exposes_and_resolves_dynamic_parameters():
    model = _floating_rho_model()
    assert {parameter.name for parameter in model.parameters} == {
        "rho.mass",
        "rho.width",
        "rho.x",
        "rho.y",
    }

    sample = model.generate_phase_space(128, seed=31)
    default_values = model.intensity(sample.as_dict())
    shifted_values = model.intensity(
        sample.as_dict(),
        {
            "rho.mass": 0.790,
            "rho.width": 0.120,
            "rho.x": 1.1,
            "rho.y": -0.3,
        },
    )
    assert not jnp.allclose(default_values, shifted_values)


def test_prepared_cache_recomputes_floating_dynamics():
    model = _floating_rho_model()
    data = model.generate_phase_space(96, seed=41)
    cache = model.prepare_cache(data)

    initial = {
        "rho.mass": 0.760,
        "rho.width": 0.180,
        "rho.x": 0.8,
        "rho.y": 0.2,
    }
    shifted = {
        "rho.mass": 0.790,
        "rho.width": 0.120,
        "rho.x": 0.8,
        "rho.y": 0.2,
    }
    intensity_initial, norm_initial = cache.evaluate(initial)
    intensity_shifted, norm_shifted = cache.evaluate(shifted)
    assert not jnp.allclose(intensity_initial, intensity_shifted)
    assert not jnp.allclose(norm_initial, norm_shifted)


def test_prepared_cache_matches_direct_model_for_floating_dynamics():
    model = _floating_rho_model()
    data = model.generate_phase_space(128, seed=51)
    cache = model.prepare_cache(data)
    norm = model.normalization_sample

    points = (
        {"rho.mass": 0.735, "rho.width": 0.095, "rho.x": -0.3, "rho.y": 1.1},
        {"rho.mass": 0.775, "rho.width": 0.149, "rho.x": 0.8, "rho.y": 0.2},
        {"rho.mass": 0.815, "rho.width": 0.235, "rho.x": 1.4, "rho.y": -0.7},
    )
    for values in points:
        cached_intensity, cached_norm = cache.evaluate(values)
        direct_intensity = model.intensity(data.as_dict(), values)
        direct_norm = jnp.mean(
            norm.weights * model.intensity(norm.as_dict(), values)
        )
        assert jnp.allclose(
            cached_intensity,
            direct_intensity,
            rtol=1e-11,
            atol=1e-12,
        )
        assert jnp.allclose(cached_norm, direct_norm, rtol=1e-11, atol=1e-12)


def test_decay_model_builds_normalized_pdf_with_internal_grid():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(channel, [NonResonant(RealImag(1.0, 0.0))])
    pdf = model.pdf()
    norm = model.normalization_sample
    values = pdf(norm.as_dict(), {})
    assert values.shape == (norm.size,)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(values > 0.0))


def test_amplitude_model_is_built_once_and_reused():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [
            Resonance(
                "rho_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=0.775,
                width=0.149,
                spin=1,
            ),
            NonResonant(RealImag(0.2, -0.1)),
        ],
    )
    first = model.amplitude_model
    second = model.amplitude_model
    assert first is second
    assert first.components[0] is second.components[0]


def test_compact_prepare_kernel_is_reused_by_model():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = _model(
        channel,
        [NonResonant(RealImag(1.0, 0.0), name="NR")],
    )
    first = model._compact_prepare_kernel(
        normalize_components=True,
        has_efficiency=False,
    )
    second = model._compact_prepare_kernel(
        normalize_components=True,
        has_efficiency=False,
    )
    efficient = model._compact_prepare_kernel(
        normalize_components=True,
        has_efficiency=True,
    )
    assert first is second
    assert efficient is not first

    first_data = model.generate_phase_space(32, seed=101)
    second_data = model.generate_phase_space(32, seed=102)
    first_cache = model.prepare_cache(first_data)
    second_cache = model.prepare_cache(second_data)
    assert first_cache.is_compact
    assert second_cache.is_compact
    assert len(model._compact_prepare_kernels) == 2


def test_auto_normalization_detects_symmetrized_narrow_bands():
    channel = DecayChannel("B+", ("pi+", "pi+", "pi-"))
    model = DecayModel(
        channel,
        [
            Resonance(
                "omega782",
                pair=(0, 2),
                coefficient=RealImag(0.09, -0.01),
                mass=0.78265,
                width=0.00849,
                spin=1,
            )
        ],
        normalization_method="auto",
        normalization_bin_width=0.05,
        normalization_binning_factor=5.0,
    )

    scheme = model.auto_normalization_scheme
    assert scheme["mode"] == "adaptive-gauss-legendre"
    assert scheme["m13_narrow_resonances"] == ((0.78265, 0.00849),)
    assert scheme["m23_narrow_resonances"] == ((0.78265, 0.00849),)
    assert any(segment.narrow for segment in scheme["m13_segments"])
    assert any(segment.narrow for segment in scheme["m23_segments"])


def test_auto_normalization_switches_to_square_dp_for_diagonal_narrow_band():
    channel = DecayChannel("D+", ("pi-", "pi+", "K+"))
    model = DecayModel(
        channel,
        [
            Resonance(
                "narrow_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=0.77,
                width=0.010,
                spin=0,
            )
        ],
        normalization_method="auto",
        normalization_resolution=120,
    )

    scheme = model.auto_normalization_scheme
    assert scheme["mode"] == "square-dalitz"
    assert scheme["pair"] == (0, 1)
    assert scheme["resolution"] == 120
    assert scheme["m12_narrow_resonances"] == ((0.77, 0.01),)


def test_auto_normalization_uses_nominal_values_for_floating_dynamics():
    channel = DecayChannel("D+", ("pi-", "K+", "pi+"))
    mass = Parameter.dynamics(
        "narrow.mass",
        0.90,
        owner="narrow",
        backend_name="mass",
        bounds=(0.85, 0.95),
    )
    width = Parameter.dynamics(
        "narrow.width",
        0.010,
        owner="narrow",
        backend_name="width",
        bounds=(0.005, 0.030),
    )
    model = DecayModel(
        channel,
        [
            Resonance(
                "narrow",
                pair=(0, 2),
                coefficient=RealImag(1.0, 0.0),
                mass=mass,
                width=width,
                spin=0,
            )
        ],
        normalization_method="auto",
        normalization_bin_width=0.05,
        normalization_binning_factor=5.0,
    )

    scheme = model.auto_normalization_scheme
    assert scheme["m13_narrow_resonances"] == ((0.9, 0.01),)
