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
    return DecayModel(
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


def test_decay_model_rejects_duplicate_component_names():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    with pytest.raises(ValueError, match="component names must be unique"):
        DecayModel(
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
        DecayModel(
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
    model = DecayModel(
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
    model = DecayModel(
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
    norm = model.generate_phase_space(256, seed=42)
    cache = model.prepare_cache(data, norm)

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
    norm = model.generate_phase_space(1024, seed=52)
    cache = model.prepare_cache(data, norm)

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


def test_decay_model_builds_normalized_pdf():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = DecayModel(channel, [NonResonant(RealImag(1.0, 0.0))])
    norm = model.generate_phase_space(512, seed=21)
    pdf = model.pdf(norm)
    values = pdf(norm.as_dict(), {})
    assert values.shape == (512,)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(values > 0.0))
