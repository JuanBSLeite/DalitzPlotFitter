import jax
import jax.numpy as jnp

from dalitzplotfitter.kinematics import (
    ThreeBodyPhaseSpace,
    invariant_mass_squared,
)


def test_reconstructed_momenta_match_dalitz_invariants():
    mother_mass = 1.86966
    pion_mass = 0.13957
    phase_space = ThreeBodyPhaseSpace(
        mother_mass,
        (pion_mass, pion_mass, pion_mass),
    )
    sample = phase_space.generate(jax.random.key(42), 2048)

    p1, p2, p3 = sample.p1, sample.p2, sample.p3
    assert p1 is not None and p2 is not None and p3 is not None

    total = p1 + p2 + p3
    assert jnp.allclose(total[:, 0], mother_mass, atol=2e-6)
    assert jnp.allclose(total[:, 1:], 0.0, atol=2e-6)

    assert jnp.allclose(invariant_mass_squared(p1 + p2), sample.s12, atol=2e-6)
    assert jnp.allclose(invariant_mass_squared(p1 + p3), sample.s13, atol=2e-6)
    assert jnp.allclose(invariant_mass_squared(p2 + p3), sample.s23, atol=2e-6)

    expected_mass_sq = pion_mass**2
    assert jnp.allclose(invariant_mass_squared(p1), expected_mass_sq, atol=2e-6)
    assert jnp.allclose(invariant_mass_squared(p2), expected_mass_sq, atol=2e-6)
    assert jnp.allclose(invariant_mass_squared(p3), expected_mass_sq, atol=2e-6)
