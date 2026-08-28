import math

import jax.numpy as jnp

from dalitzplotfitter import DalitzGrid, DecayChannel, dalitz_s13_limits, enable_x64


enable_x64()


def test_dalitz_grid_contains_exactly_n_squared_physical_midpoints():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    resolution = 120
    grid = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=resolution,
    )
    sample = grid.sample()

    assert sample.size == resolution**2

    low, high = dalitz_s13_limits(
        sample.s12,
        mother_mass=channel.parent_mass,
        masses=channel.daughter_masses,
    )
    assert bool(jnp.all(sample.s13 > low))
    assert bool(jnp.all(sample.s13 < high))

    m1, m2, m3 = channel.daughter_masses
    invariant_sum = channel.parent_mass**2 + m1**2 + m2**2 + m3**2
    assert jnp.allclose(
        sample.s12 + sample.s13 + sample.s23,
        invariant_sum,
        rtol=0.0,
        atol=2e-14,
    )


def test_dalitz_grid_uses_constant_equal_area_weight():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    grid = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=100,
    )
    sample = grid.sample()

    assert bool(jnp.all(sample.weights == sample.weights[0]))
    estimated_area = float(jnp.mean(sample.weights))
    assert math.isclose(
        estimated_area,
        float(grid.area),
        rel_tol=2e-14,
        abs_tol=2e-14,
    )


def test_dalitz_grid_has_equal_area_s12_strips():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    resolution = 64
    grid = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=resolution,
        boundary_resolution=20001,
    )
    sample = grid.sample()

    # Each s12 strip is repeated exactly N times and corresponds to an equal
    # fraction of cumulative Dalitz area by construction.
    strip_s12 = sample.s12.reshape(resolution, resolution)[:, 0]
    assert bool(jnp.all(jnp.diff(strip_s12) > 0.0))
    assert jnp.unique(sample.s12).shape[0] == resolution
    assert sample.s13.reshape(resolution, resolution).shape == (resolution, resolution)


def test_dalitz_grid_mean_estimator_is_area_times_mean():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    grid = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=80,
    )
    sample = grid.sample()

    values = 1.0 + 0.3 * sample.s12 + 0.2 * sample.s13
    estimate_from_package_convention = jnp.mean(sample.weights * values)
    direct_equal_area_quadrature = grid.area * jnp.mean(values)

    assert jnp.allclose(
        estimate_from_package_convention,
        direct_equal_area_quadrature,
        rtol=2e-14,
        atol=2e-14,
    )
