import jax
import jax.numpy as jnp

from dalitzplotfitter.kinematics import ThreeBodyPhaseSpace, inside_dalitz, invariant_sum


def test_generated_points_are_inside_dalitz_region():
    phase_space = ThreeBodyPhaseSpace(
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
    )
    sample = phase_space.generate(jax.random.key(1234), 10_000)
    mask = inside_dalitz(
        sample.s12,
        sample.s23,
        phase_space.mother_mass,
        phase_space.masses,
    )
    assert bool(jnp.all(mask))
    assert bool(jnp.all(sample.weights >= 0.0))


def test_invariant_sum_rule():
    phase_space = ThreeBodyPhaseSpace(1.86966, (0.13957, 0.13957, 0.13957))
    sample = phase_space.generate(jax.random.key(7), 1000)
    expected = invariant_sum(phase_space.mother_mass, phase_space.masses)
    assert jnp.allclose(sample.s12 + sample.s13 + sample.s23, expected)
