import jax.numpy as jnp

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    NonResonant,
    Parameter,
    RealImag,
    Resonance,
)


class ConstantLineshape:
    """Minimal test plugin proving DecayModel is lineshape-agnostic."""

    def __call__(self, mass, context):
        del context
        return jnp.ones_like(mass, dtype=jnp.complex128)


def test_decay_channel_resolves_particle_masses_in_gev():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    assert 1.86 < channel.parent_mass < 1.88
    assert all(0.139 < mass < 0.141 for mass in channel.daughter_masses)


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
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    mass = Parameter.dynamics(
        "rho.mass",
        0.760,
        owner="rho",
        backend_name="mass",
        bounds=(0.72, 0.82),
    )
    width = Parameter.dynamics(
        "rho.width",
        0.180,
        owner="rho",
        backend_name="width",
        bounds=(0.08, 0.25),
    )
    x = Parameter.coefficient("rho.x", 0.8, owner="rho")
    y = Parameter.coefficient("rho.y", 0.2, owner="rho")
    model = DecayModel(
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
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    mass = Parameter.dynamics(
        "rho.mass",
        0.760,
        owner="rho",
        backend_name="mass",
        bounds=(0.72, 0.82),
    )
    width = Parameter.dynamics(
        "rho.width",
        0.180,
        owner="rho",
        backend_name="width",
        bounds=(0.08, 0.25),
    )
    x = Parameter.coefficient("rho.x", 0.8, owner="rho")
    y = Parameter.coefficient("rho.y", 0.2, owner="rho")
    model = DecayModel(
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


def test_decay_model_builds_normalized_pdf():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = DecayModel(channel, [NonResonant(RealImag(1.0, 0.0))])
    norm = model.generate_phase_space(512, seed=21)
    pdf = model.pdf(norm)
    values = pdf(norm.as_dict(), {})
    assert values.shape == (512,)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(values > 0.0))
