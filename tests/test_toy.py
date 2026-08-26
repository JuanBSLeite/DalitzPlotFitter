import jax
import jax.numpy as jnp

from dalitzplotfitter import ThreeBodyPhaseSpace, enable_x64
from dalitzplotfitter.toy import ToyGenerator


enable_x64()


def _energy_transform(momentum):
    return {"energy": momentum["p0"][:, 0]}


def _constant_intensity(data, parameters):
    del parameters
    return jnp.ones_like(data["energy"])


def test_accept_reject_generates_unweighted_continuous_phase_space():
    phase_space = ThreeBodyPhaseSpace(
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
    )
    generator = ToyGenerator(
        phase_space=phase_space,
        transformer=_energy_transform,
        pool_size=10_000,
        batch_size=4_000,
        envelope_safety=1.3,
    )

    toy, data = generator.generate(
        jax.random.key(2026),
        size=2_000,
        intensity=_constant_intensity,
        parameters={},
    )

    assert toy.size == 2_000
    assert data["energy"].shape == (2_000,)
    assert jnp.allclose(toy.weights, jnp.ones(2_000))
    assert bool(jnp.all(jnp.isfinite(toy.s12)))
    assert bool(jnp.all(jnp.isfinite(toy.s13)))
    assert bool(jnp.all(jnp.isfinite(toy.s23)))

    # Accept-reject generates fresh continuous candidates rather than drawing
    # repeatedly from a finite categorical pool. Exact duplicate s12 values
    # should therefore be absent for this deterministic test seed.
    assert int(jnp.unique(toy.s12).shape[0]) == toy.size


def test_accept_reject_includes_phase_space_proposal_weight():
    phase_space = ThreeBodyPhaseSpace(
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
    )
    generator = ToyGenerator(
        phase_space=phase_space,
        transformer=_energy_transform,
        pool_size=20_000,
        batch_size=5_000,
        envelope_safety=1.3,
    )

    toy, _ = generator.generate(
        jax.random.key(7),
        size=5_000,
        intensity=_constant_intensity,
        parameters={},
    )

    # For constant matrix element the physical Dalitz density is uniform in
    # ds12 ds23. The native proposal is not uniform in that measure, so its
    # Jacobian weights are required. Compare the unweighted toy mean with an
    # independent weighted proposal estimate of the same phase-space mean.
    reference = phase_space.generate(
        jax.random.key(8),
        size=100_000,
        with_momenta=False,
    )
    expected_mean = jnp.sum(reference.weights * reference.s12) / jnp.sum(
        reference.weights
    )
    assert jnp.isclose(jnp.mean(toy.s12), expected_mean, rtol=0.0, atol=0.025)
