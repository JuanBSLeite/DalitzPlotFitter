import jax.numpy as jnp

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    NonResonant,
    RealImag,
    Resonance,
)


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


def test_decay_model_builds_normalized_pdf():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = DecayModel(channel, [NonResonant(RealImag(1.0, 0.0))])
    norm = model.generate_phase_space(512, seed=21)
    pdf = model.pdf(norm)
    values = pdf(norm.as_dict(), {})
    assert values.shape == (512,)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(values > 0.0))
