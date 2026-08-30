import math

import jax.numpy as jnp

from dalitzplotfitter import DalitzGrid, DalitzMC, DecayChannel, dalitz_s13_limits, enable_x64


enable_x64()


def test_dalitz_mc_is_uniform_area_sample_with_constant_weight():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    mc = DalitzMC(channel.parent_mass, channel.daughter_masses)
    sample = mc.generate(100_000, seed=2027)

    assert sample.size == 100_000
    assert bool(jnp.all(jnp.isfinite(sample.weights)))
    assert bool(jnp.all(sample.weights == sample.weights[0]))

    low, high = dalitz_s13_limits(
        sample.s12,
        mother_mass=channel.parent_mass,
        masses=channel.daughter_masses,
    )
    assert bool(jnp.all(sample.s13 >= low))
    assert bool(jnp.all(sample.s13 <= high))

    reference_area = float(
        DalitzGrid(
            channel.parent_mass,
            channel.daughter_masses,
            resolution=100,
            boundary_resolution=20001,
        ).area
    )
    assert math.isclose(
        float(jnp.mean(sample.weights)),
        reference_area,
        rel_tol=2e-14,
        abs_tol=2e-14,
    )


def test_dalitz_mc_is_reproducible_for_fixed_seed():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    mc = DalitzMC(channel.parent_mass, channel.daughter_masses)
    a = mc.generate(1000, seed=7)
    b = mc.generate(1000, seed=7)

    assert jnp.array_equal(a.s12, b.s12)
    assert jnp.array_equal(a.s13, b.s13)
    assert jnp.array_equal(a.s23, b.s23)
