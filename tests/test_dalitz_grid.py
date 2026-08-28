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
    invariant_sum = (
        channel.parent_mass**2 + m1**2 + m2**2 + m3**2
    )
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
    assert jnp.all(sample.weights == sample.weights[0])

    m1, m2, m3 = channel.daughter_masses
    ds12 = (
        (channel.parent_mass - m3) ** 2 - (m1 + m2) ** 2
    ) / resolution
    ds13 = (
        (channel.parent_mass - m2) ** 2 - (m1 + m3) ** 2
    ) / resolution
    expected_area = sample.size * ds12 * ds13

    # The package integration convention is mean(weights * f). For f=1 this
    # must equal the midpoint estimate of the physical Dalitz area.
    estimated_area = float(jnp.mean(sample.weights))
    assert math.isclose(estimated_area, expected_area, rel_tol=2e-14, abs_tol=2e-14)


def test_dalitz_grid_area_converges_with_resolution():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    areas = []
    for resolution in (100, 200, 400):
        sample = DalitzGrid(
            channel.parent_mass,
            channel.daughter_masses,
            resolution=resolution,
        ).sample()
        areas.append(float(jnp.mean(sample.weights)))

    # The boundary error of midpoint masking must shrink. Do not encode an
    # external analytic area value; compare successive deterministic grids.
    assert abs(areas[2] - areas[1]) < abs(areas[1] - areas[0])
