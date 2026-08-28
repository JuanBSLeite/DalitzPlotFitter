import math

import jax.numpy as jnp

from dalitzplotfitter import DalitzGrid, DecayChannel, dalitz_s13_limits, enable_x64


enable_x64()


def test_dalitz_grid_contains_only_physical_midpoints():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    sample = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=120,
    ).sample()

    low, high = dalitz_s13_limits(
        sample.s12,
        mother_mass=channel.parent_mass,
        masses=channel.daughter_masses,
    )
    assert bool(jnp.all(sample.s13 >= low))
    assert bool(jnp.all(sample.s13 <= high))

    m1, m2, m3 = channel.daughter_masses
    invariant_sum = channel.parent_mass**2 + m1**2 + m2**2 + m3**2
    assert jnp.allclose(
        sample.s12 + sample.s13 + sample.s23,
        invariant_sum,
        rtol=0.0,
        atol=2e-14,
    )


def test_dalitz_grid_uses_constant_midpoint_weights():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    resolution = 100
    sample = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=resolution,
    ).sample()

    assert sample.size > 0
    assert bool(jnp.all(sample.weights == sample.weights[0]))

    m1, m2, m3 = channel.daughter_masses
    ds12 = ((channel.parent_mass - m3) ** 2 - (m1 + m2) ** 2) / resolution
    ds13 = ((channel.parent_mass - m2) ** 2 - (m1 + m3) ** 2) / resolution
    expected_area = sample.size * ds12 * ds13

    # The package integration convention is mean(weights * f). For f=1 this
    # must exactly reproduce the midpoint quadrature cell_area * N_valid.
    estimated_area = float(jnp.mean(sample.weights))
    assert math.isclose(estimated_area, expected_area, rel_tol=2e-14, abs_tol=2e-14)


def test_dalitz_grid_mean_estimator_equals_midpoint_sum():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    resolution = 80
    sample = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=resolution,
    ).sample()

    values = 1.0 + 0.3 * sample.s12 + 0.2 * sample.s13
    estimate_from_package_convention = jnp.mean(sample.weights * values)

    m1, m2, m3 = channel.daughter_masses
    ds12 = ((channel.parent_mass - m3) ** 2 - (m1 + m2) ** 2) / resolution
    ds13 = ((channel.parent_mass - m2) ** 2 - (m1 + m3) ** 2) / resolution
    direct_midpoint_sum = ds12 * ds13 * jnp.sum(values)

    assert jnp.allclose(
        estimate_from_package_convention,
        direct_midpoint_sum,
        rtol=2e-14,
        atol=2e-14,
    )
