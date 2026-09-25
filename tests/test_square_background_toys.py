"""Distribution closure for backgrounds defined per Square-Dalitz area."""

import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    CPToyBackground,
    DecayChannel,
    DecayModel,
    NonResonant,
    RealImag,
    SquareDalitzHistogramBackground,
    ToyBackground,
    generate_cp_toy,
    generate_toy,
    prepare_inverse_toy_generator,
)


def setup_background(*, folded=False, values=None, divide_jacobian=True):
    channel = DecayChannel("B+", ("pi-", "pi+", "pi+"))
    model = DecayModel(
        channel,
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_pair=(1, 2),
        normalization_resolution=80,
    )
    shape = SquareDalitzHistogramBackground(
        jnp.linspace(0, 1, 5),
        jnp.linspace(0, 0.5 if folded else 1, 5),
        jnp.ones((4, 4)) if values is None else jnp.asarray(values),
        channel.parent_mass,
        channel.daughter_masses,
        pair=(1, 2),
        folded=folded,
        divide_jacobian=divide_jacobian,
    )
    return model, shape


def assert_histogram(toy, shape, expected):
    mp, tp = shape.square_coordinates(toy.as_dict())
    counts = np.histogram2d(
        np.asarray(mp),
        np.asarray(tp),
        bins=(np.asarray(shape.mprime_edges), np.asarray(shape.thetaprime_edges)),
    )[0]
    expected = np.asarray(expected, dtype=float)
    expected /= expected.sum()
    # Six binomial standard deviations plus a small CDF grid allowance.
    tolerance = 6 * np.sqrt(expected * (1 - expected) / toy.size) + 0.003
    assert np.all(np.abs(counts / toy.size - expected) < tolerance)


@pytest.mark.parametrize("method", ["accept-reject", "inverse-transform", "prepared"])
@pytest.mark.parametrize("folded", [False, True])
def test_square_background_distribution(method, folded):
    values = np.arange(1, 17).reshape(4, 4)
    model, shape = setup_background(folded=folded, values=values)
    kwargs = dict(signal_fraction=0, backgrounds=[ToyBackground("bkg", shape)])
    if method == "prepared":
        toy = prepare_inverse_toy_generator(model, resolution=256, **kwargs).generate(
            20_000,
            seed=821,
            include_momenta=False,
        )
    else:
        toy = generate_toy(
            model,
            20_000,
            method=method,
            seed=821,
            include_momenta=False,
            inverse_resolution=256,
            **kwargs,
        )
    assert toy.p1 is None
    assert_histogram(toy, shape, values)


@pytest.mark.parametrize("method", ["accept-reject", "inverse-transform"])
def test_square_background_flat_including_edges(method):
    model, shape = setup_background()
    toy = generate_toy(
        model,
        20_000,
        method=method,
        signal_fraction=0,
        backgrounds=[ToyBackground("bkg", shape)],
        seed=912,
        inverse_resolution=128,
        include_momenta=True,
    )
    assert toy.p1 is not None
    assert np.all(np.isfinite(toy.p1))
    assert_histogram(toy, shape, np.ones((4, 4)))


@pytest.mark.parametrize("method", ["accept-reject", "inverse-transform"])
@pytest.mark.parametrize("scaled", [False, True])
def test_cp_background_charge_split_and_veto(method, scaled):
    # Equal square integrals but substantially different Dalitz-volume integrals.
    plus_values = np.tile([1, 2, 3, 4], (4, 1)).T
    minus_values = plus_values[::-1].copy()
    model, plus = setup_background(values=plus_values)
    _, minus = setup_background(values=minus_values)

    def veto(data):
        return plus.square_coordinates(data)[1] < 0.5

    plus_shape, minus_shape = plus, minus
    expected_plus = 0.5
    if scaled:
        plus_shape = plus.with_charge_asymmetry(
            model.normalization_sample,
            charge=1,
            asymmetry=0.4,
            veto=veto,
        )
        minus_shape = minus.with_charge_asymmetry(
            model.normalization_sample,
            charge=-1,
            asymmetry=0.4,
            veto=veto,
        )
        expected_plus = 0.3
    toys = generate_cp_toy(
        model,
        model,
        30_000,
        method=method,
        signal_fraction=0,
        backgrounds=[CPToyBackground("bkg", plus_shape, minus_shape)],
        plus_veto=veto,
        minus_veto=veto,
        seed=431,
        inverse_resolution=256,
        include_momenta=False,
    )
    assert abs(toys[0].size / 30_000 - expected_plus) < 0.015
    for toy, shape, values in zip(
        toys, (plus, minus), (plus_values, minus_values), strict=True
    ):
        assert np.all(veto(toy.as_dict()))
        expected = values.copy()
        expected[:, 2:] = 0
        assert_histogram(toy, shape, expected)


@pytest.mark.parametrize("method", ["accept-reject", "inverse-transform"])
def test_unconverted_background_keeps_dalitz_measure(method):
    model, shape = setup_background(divide_jacobian=False)
    toy = generate_toy(
        model,
        20_000,
        method=method,
        signal_fraction=0,
        backgrounds=[ToyBackground("bkg", shape)],
        seed=814,
        inverse_resolution=256,
        include_momenta=False,
    )
    # Without conversion a constant callable means uniform conventional Dalitz.
    mp, _ = shape.square_coordinates(toy.as_dict())
    assert np.mean((np.asarray(mp) > 0.25) & (np.asarray(mp) < 0.75)) > 0.8


def test_square_background_accept_reject_preserves_momentum_veto():
    model, shape = setup_background()

    def veto(data):
        return data["p1"][:, 3] > 0

    toy = generate_toy(
        model,
        1000,
        method="accept-reject",
        signal_fraction=0,
        backgrounds=[ToyBackground("bkg", shape)],
        veto=veto,
        seed=821,
    )
    assert np.all(veto(toy.as_dict()))


@pytest.mark.parametrize("method", ["accept-reject", "inverse-transform"])
@pytest.mark.parametrize("interpolation,pair", [("linear", (0, 2)), ("spline", (2, 1))])
def test_interpolated_square_background_and_ordered_pair(method, interpolation, pair):
    from dataclasses import replace

    from dalitzplotfitter import square_dalitz_to_invariants

    model, shape = setup_background(values=np.arange(1, 17).reshape(4, 4))
    shape = replace(shape, interpolation=interpolation, pair=pair)
    # Independent midpoint integration in square coordinates, with equal-area
    # weights: no physical Dalitz Jacobian belongs in these expected counts.
    grid = (np.arange(400) + 0.5) / 400
    mp, tp = np.meshgrid(grid, grid, indexing="ij")
    inv = square_dalitz_to_invariants(
        mp.ravel(),
        tp.ravel(),
        mother_mass=shape.mother_mass,
        masses=shape.masses,
        pair=pair,
    )
    heights = shape.generation_value(dict(zip(("s12", "s13", "s23"), inv, strict=True)))
    expected = np.histogram2d(
        mp.ravel(),
        tp.ravel(),
        bins=(shape.mprime_edges, shape.thetaprime_edges),
        weights=np.asarray(heights),
    )[0]
    toy = generate_toy(
        model,
        20_000,
        method=method,
        signal_fraction=0,
        backgrounds=[ToyBackground("bkg", shape)],
        seed=416,
        inverse_resolution=256,
        include_momenta=False,
    )
    assert_histogram(toy, shape, expected)
