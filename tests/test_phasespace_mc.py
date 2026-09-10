import jax
import jax.numpy as jnp

from dalitzplotfitter import enable_x64
from dalitzplotfitter.kinematics import PhaseSpaceMC, invariant_mass_squared


enable_x64()


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


def test_compact_phase_space_is_physical_without_building_momenta():
    mother_mass = 1.86966
    masses = (0.13957, 0.13957, 0.13957)
    generator = PhaseSpaceMC(mother_mass, masses)
    sample = generator.generate(2048, seed=29, include_momenta=False)

    assert sample.p1 is None and sample.p2 is None and sample.p3 is None
    assert sample.size == 2048
    assert bool(jnp.all(jnp.isfinite(sample.weights)))
    assert bool(jnp.all(sample.weights > 0.0))
    invariant_sum = sample.s12 + sample.s13 + sample.s23
    expected = mother_mass**2 + sum(mass**2 for mass in masses)
    assert jnp.allclose(invariant_sum, expected, rtol=0.0, atol=1e-12)


def test_attach_momenta_reconstructs_compact_invariants():
    mother_mass = 1.86966
    masses = (0.13957, 0.13957, 0.13957)
    generator = PhaseSpaceMC(mother_mass, masses)
    compact = generator.generate(1024, seed=31, include_momenta=False)
    full = generator.attach_momenta(compact, seed=32)

    assert jnp.array_equal(full.s12, compact.s12)
    assert jnp.array_equal(full.s13, compact.s13)
    assert jnp.array_equal(full.s23, compact.s23)
    assert jnp.array_equal(full.weights, compact.weights)

    total = full.p1 + full.p2 + full.p3
    assert jnp.allclose(total[:, 0], mother_mass, rtol=0.0, atol=1e-9)
    assert jnp.allclose(total[:, 1:], 0.0, rtol=0.0, atol=1e-9)
    assert jnp.allclose(
        invariant_mass_squared(full.p1 + full.p2),
        compact.s12,
        rtol=1e-9,
        atol=1e-10,
    )
    assert jnp.allclose(
        invariant_mass_squared(full.p1 + full.p3),
        compact.s13,
        rtol=1e-9,
        atol=1e-10,
    )
    assert jnp.allclose(
        invariant_mass_squared(full.p2 + full.p3),
        compact.s23,
        rtol=1e-9,
        atol=1e-10,
    )
