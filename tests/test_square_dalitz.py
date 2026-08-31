import jax.numpy as jnp

from dalitzplotfitter import (
    DalitzGrid,
    DecayChannel,
    SquareDalitzGrid,
    enable_x64,
    invariants_to_square_dalitz,
    square_dalitz_to_invariants,
)


enable_x64()


def test_square_dalitz_round_trip_for_kpipi_pair_13():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    grid = SquareDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=25,
        pair=(0, 2),
    ).sample()

    mp, tp = invariants_to_square_dalitz(
        grid.s12,
        grid.s13,
        grid.s23,
        mother_mass=channel.parent_mass,
        masses=channel.daughter_masses,
        pair=(0, 2),
    )
    s12, s13, s23 = square_dalitz_to_invariants(
        mp,
        tp,
        mother_mass=channel.parent_mass,
        masses=channel.daughter_masses,
        pair=(0, 2),
    )

    assert bool(jnp.allclose(s12, grid.s12, rtol=1e-10, atol=1e-10))
    assert bool(jnp.allclose(s13, grid.s13, rtol=1e-10, atol=1e-10))
    assert bool(jnp.allclose(s23, grid.s23, rtol=1e-10, atol=1e-10))
    assert bool(jnp.all((mp > 0.0) & (mp < 1.0)))
    assert bool(jnp.all((tp > 0.0) & (tp < 1.0)))


def test_square_dalitz_constant_integral_matches_dalitz_area_midpoint():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    dalitz = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=220,
    ).sample()
    square = SquareDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=220,
        pair=(0, 2),
        quadrature="midpoint",
    ).sample()

    direct = jnp.mean(dalitz.weights)
    transformed = jnp.mean(square.weights)
    assert jnp.allclose(transformed, direct, rtol=2e-4, atol=1e-6)


def test_square_dalitz_gauss_constant_integral_matches_dalitz_area():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    dalitz = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=500,
    ).sample()
    square = SquareDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=80,
        pair=(0, 2),
        quadrature="gauss-legendre",
    ).sample()

    direct = jnp.mean(dalitz.weights)
    transformed = jnp.mean(square.weights)
    assert jnp.allclose(transformed, direct, rtol=2e-5, atol=1e-7)


def test_square_dalitz_gauss_smooth_integrals_match_dalitz_grid():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    dalitz = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=500,
    ).sample()
    square = SquareDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=100,
        pair=(0, 2),
        quadrature="gauss-legendre",
    ).sample()

    functions = (
        lambda sample: sample.s13,
        lambda sample: sample.s23,
        lambda sample: sample.s13 * sample.s23,
    )
    for function in functions:
        first = jnp.mean(dalitz.weights * function(dalitz))
        second = jnp.mean(square.weights * function(square))
        assert jnp.allclose(second, first, rtol=2e-4, atol=1e-6)


def test_square_dalitz_gauss_narrow_structure_matches_dense_reference():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))

    def narrow(sample):
        return 1.0 / ((sample.s13 - 0.895**2) ** 2 + (0.895 * 0.047) ** 2)

    reference = SquareDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=500,
        pair=(0, 2),
        quadrature="gauss-legendre",
    ).sample()
    square = SquareDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=120,
        pair=(0, 2),
        quadrature="gauss-legendre",
    ).sample()

    expected = jnp.mean(reference.weights * narrow(reference))
    transformed = jnp.mean(square.weights * narrow(square))
    assert jnp.allclose(transformed, expected, rtol=2e-4, atol=1e-6)
