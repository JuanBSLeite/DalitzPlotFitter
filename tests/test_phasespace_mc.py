import jax
import jax.numpy as jnp

from dalitzplotfitter.kinematics import PhaseSpaceMC, invariant_mass_squared


def test_phase_space_mc_returns_weighted_conserving_three_body_events():
    mother_mass = 1.86966
    pion_mass = 0.13957
    generator = PhaseSpaceMC(
        mother_mass=mother_mass,
        masses=(pion_mass, pion_mass, pion_mass),
    )
    sample = generator.generate(512, seed=2026)

    assert sample.size == 512
    assert sample.weights.shape == (512,)
    assert bool(jnp.all(jnp.isfinite(sample.weights)))
    assert bool(jnp.all(sample.weights > 0.0))

    total = sample.p1 + sample.p2 + sample.p3
    assert jnp.allclose(total[:, 0], mother_mass, rtol=0.0, atol=1e-9)
    assert jnp.allclose(total[:, 1:], 0.0, rtol=0.0, atol=1e-9)

    for momentum in (sample.p1, sample.p2, sample.p3):
        assert jnp.allclose(
            invariant_mass_squared(momentum),
            pion_mass**2,
            rtol=1e-8,
            atol=1e-10,
        )

    invariant_sum = sample.s12 + sample.s13 + sample.s23
    expected = mother_mass**2 + 3 * pion_mass**2
    assert jnp.allclose(invariant_sum, expected, rtol=1e-8, atol=1e-9)


def test_phase_space_mc_is_reproducible_with_seed_or_key():
    generator = PhaseSpaceMC(1.86966, (0.13957, 0.13957, 0.13957))
    first = generator.generate(64, seed=17)
    second = generator.generate(64, seed=17)
    keyed = generator.generate(64, key=jax.random.key(17))

    for name in ("s12", "s13", "s23", "weights", "p1", "p2", "p3"):
        assert jnp.array_equal(getattr(first, name), getattr(second, name))
        assert jnp.array_equal(getattr(first, name), getattr(keyed, name))


def test_phase_space_mc_rejects_seed_and_key_together():
    generator = PhaseSpaceMC(1.86966, (0.13957, 0.13957, 0.13957))
    try:
        generator.generate(8, seed=1, key=jax.random.key(1))
    except ValueError as error:
        assert "either seed or key" in str(error)
    else:
        raise AssertionError("expected ValueError")
