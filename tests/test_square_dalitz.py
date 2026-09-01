import jax.numpy as jnp
import numpy as np

from dalitzplotfitter import (
    DalitzGaussLegendreGrid,
    DecayChannel,
    SquareDalitzGrid,
    enable_x64,
    invariants_to_square_dalitz,
    square_dalitz_jacobian,
    square_dalitz_to_invariants,
)

enable_x64()


def test_square_dalitz_defaults_match_laura_convention():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    grid = SquareDalitzGrid(channel.parent_mass, channel.daughter_masses)

    assert grid.resolution == 1000
    assert grid.pair == (0, 1)
    assert grid.quadrature == "gauss-legendre"


def test_reversing_ordered_pair_reflects_laura_helicity_angle():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    mp = jnp.asarray([0.18, 0.43, 0.76])
    tp = jnp.asarray([0.12, 0.39, 0.81])
    invariants = square_dalitz_to_invariants(
        mp,
        tp,
        mother_mass=channel.parent_mass,
        masses=channel.daughter_masses,
        pair=(0, 1),
    )

    reversed_mp, reversed_tp = invariants_to_square_dalitz(
        *invariants,
        mother_mass=channel.parent_mass,
        masses=channel.daughter_masses,
        pair=(1, 0),
    )

    assert jnp.allclose(reversed_mp, mp, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(reversed_tp, 1.0 - tp, rtol=1e-12, atol=1e-12)


def test_square_dalitz_jacobian_matches_laura_factorization():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    masses = channel.daughter_masses
    mp = np.asarray([0.16, 0.37, 0.68, 0.89])
    tp = np.asarray([0.11, 0.42, 0.63, 0.84])
    mi, mj, mk = masses
    delta_m = channel.parent_mass - mk - mi - mj
    m12 = mi + mj + 0.5 * delta_m * (1.0 + np.cos(np.pi * mp))
    s12 = m12**2
    kallen_12 = s12**2 + mi**4 + mj**4 - 2.0 * (
        s12 * mi**2 + s12 * mj**2 + mi**2 * mj**2
    )
    kallen_b = channel.parent_mass**4 + s12**2 + mk**4 - 2.0 * (
        channel.parent_mass**2 * s12
        + channel.parent_mass**2 * mk**2
        + s12 * mk**2
    )
    q = np.sqrt(kallen_12) / (2.0 * m12)
    p = np.sqrt(kallen_b) / (2.0 * m12)
    dm12_dmp = 0.5 * np.pi * delta_m * np.sin(np.pi * mp)
    dcostheta_dtp = np.pi * np.sin(np.pi * tp)
    expected = 4.0 * p * q * m12 * dm12_dmp * dcostheta_dtp

    actual = square_dalitz_jacobian(
        mp,
        tp,
        mother_mass=channel.parent_mass,
        masses=masses,
        pair=(0, 1),
    )
    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)


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
    reference = DalitzGaussLegendreGrid(
        channel.parent_mass,
        channel.daughter_masses,
        order_m13=300,
        order_m23=300,
    ).sample()
    square = SquareDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=220,
        pair=(0, 2),
        quadrature="midpoint",
    ).sample()

    direct = jnp.mean(reference.weights)
    transformed = jnp.mean(square.weights)
    assert jnp.allclose(transformed, direct, rtol=2e-4, atol=1e-6)


def test_square_dalitz_gauss_constant_integral_matches_dalitz_area():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    reference = DalitzGaussLegendreGrid(
        channel.parent_mass,
        channel.daughter_masses,
        order_m13=300,
        order_m23=300,
    ).sample()
    square = SquareDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=80,
        pair=(0, 2),
        quadrature="gauss-legendre",
    ).sample()

    direct = jnp.mean(reference.weights)
    transformed = jnp.mean(square.weights)
    assert jnp.allclose(transformed, direct, rtol=1e-4, atol=1e-7)


def test_square_dalitz_gauss_smooth_integrals_match_mass_plane_quadrature():
    channel = DecayChannel("B+", ("K+", "pi+", "pi-"))
    reference = DalitzGaussLegendreGrid(
        channel.parent_mass,
        channel.daughter_masses,
        order_m13=300,
        order_m23=300,
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
        first = jnp.mean(reference.weights * function(reference))
        second = jnp.mean(square.weights * function(square))
        assert jnp.allclose(second, first, rtol=2e-4, atol=1e-6)


def test_square_dalitz_gauss_narrow_structure_converges_with_resolution():
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
        resolution=250,
        pair=(0, 2),
        quadrature="gauss-legendre",
    ).sample()

    expected = jnp.mean(reference.weights * narrow(reference))
    transformed = jnp.mean(square.weights * narrow(square))
    assert jnp.allclose(transformed, expected, rtol=2e-4, atol=1e-6)
