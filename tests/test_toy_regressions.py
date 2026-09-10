"""Regression checks for toy support, angular acceptance and absent components."""

import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    CPRealImag,
    CPToyBackground,
    DecayChannel,
    DecayModel,
    NonResonant,
    RealImag,
    ToyBackground,
    generate_cp_toy,
    generate_signal_toy,
    generate_toy,
    prepare_inverse_toy_generator,
)
from dalitzplotfitter.inverse_transform import (
    DalitzInverseTransformSampler,
    _inverse_row,
)


def model(coefficient=None):
    return DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(RealImag(1.0, 0.0) if coefficient is None else coefficient)],
        normalization_method="square-dalitz",
        normalization_resolution=12,
    )


def test_inverse_cdf_does_not_bridge_plateau():
    result = _inverse_row(
        np.array([0.0, 0.5, 0.5, 1.0]), np.arange(4.0), np.array([0.49, 0.5, 0.51])
    )
    assert result[0] < 1
    assert np.all(result[1:] >= 2)
    endpoints = _inverse_row(
        np.array([0.0, 0.0, 0.5, 1.0, 1.0]), np.arange(5.0), np.array([0.0, 1.0])
    )
    np.testing.assert_array_equal(endpoints, [1.0, 3.0])


@pytest.mark.parametrize("resolution", [64, 256])
def test_inverse_support_and_relative_population(resolution):
    # Two disconnected strips; compare their populations with an independent
    # high-resolution integration of the physical Dalitz width.
    from dalitzplotfitter.inverse_transform import _s13_limits

    def support(d):
        return ((d["s12"] < 0.8) | (d["s12"] > 1.2)) & (d["s13"] > 0.6)

    m = model()
    sampler = DalitzInverseTransformSampler.prepare(
        m.channel.parent_mass,
        m.channel.daughter_masses,
        lambda d: support(d).astype(float),
        resolution=resolution,
    )
    toy = sampler.generate(20_000, seed=112, include_momenta=False)
    assert np.all(support(toy.as_dict()))
    grid = np.linspace(
        sum(m.channel.daughter_masses[:2]) ** 2,
        (m.channel.parent_mass - m.channel.daughter_masses[2]) ** 2,
        100_000,
    )
    low, high = _s13_limits(
        grid, mother_mass=m.channel.parent_mass, masses=m.channel.daughter_masses
    )
    weight = np.maximum(high - np.maximum(low, 0.6), 0) * ((grid < 0.8) | (grid > 1.2))
    expected = np.sum(weight * (grid < 0.8)) / np.sum(weight)
    assert abs(np.mean(np.asarray(toy.s12) < 0.8) - expected) < 0.02
    repeated = sampler.generate(20_000, seed=112, include_momenta=False)
    np.testing.assert_array_equal(toy.s12, repeated.s12)


@pytest.mark.parametrize("mixture", [False, True])
@pytest.mark.parametrize("include_momenta", [False, True])
def test_accept_reject_preserves_selected_momenta(mixture, include_momenta):
    m = model()
    options = dict(
        method="accept-reject",
        seed=24,
        pool_size=1024,
        include_momenta=include_momenta,
        efficiency=lambda d: d["p1"][:, 3] > 0,
    )
    if mixture:
        # Disjoint invariant ranges identify the signal after component shuffling.
        options["efficiency"] = lambda d: (d["p1"][:, 3] > 0) & (d["s12"] < 1.0)
        toy = generate_toy(
            m,
            500,
            signal_fraction=0.6,
            backgrounds=[ToyBackground("bg", lambda d: d["s12"] >= 1.0)],
            **options,
        )
    else:
        toy = generate_signal_toy(m, 500, **options)
    assert toy.size == 500
    if include_momenta:
        signal = np.asarray(toy.s12) < 1.0 if mixture else np.ones(500, dtype=bool)
        assert signal.any()
        assert np.all(np.asarray(toy.p1)[signal, 3] > 0)
    else:
        assert toy.p1 is toy.p2 is toy.p3 is None


@pytest.mark.parametrize("method", ["inverse-transform", "accept-reject"])
@pytest.mark.parametrize("delta", [-1.0, 1.0])
def test_cp_zero_rate_charge(method, delta):
    cp = CPRealImag(1.0, 0.0, delta, 0.0)
    options = (
        {"inverse_resolution": 32}
        if method == "inverse-transform"
        else {"pool_size": 512}
    )
    plus, minus = generate_cp_toy(
        model(cp.for_charge(1)),
        model(cp.for_charge(-1)),
        100,
        seed=1,
        method=method,
        include_momenta=False,
        **options,
    )
    assert (plus.size, minus.size) == ((100, 0) if delta == 1 else (0, 100))
    assert plus.p1 is minus.p1 is None


@pytest.mark.parametrize("method", ["inverse-transform", "accept-reject"])
def test_cp_invalid_joint_rate_and_pure_background(method):
    zero = model(RealImag(0.0, 0.0))
    options = (
        {"inverse_resolution": 32}
        if method == "inverse-transform"
        else {"pool_size": 512}
    )
    with pytest.raises(ValueError, match="integrals"):
        generate_cp_toy(zero, zero, 50, method=method, **options)
    plus, minus = generate_cp_toy(
        zero,
        zero,
        50,
        method=method,
        signal_fraction=0.0,
        seed=1,
        backgrounds=[
            CPToyBackground(
                "bg",
                lambda d: jnp.ones_like(d["s12"]),
                lambda d: jnp.zeros_like(d["s12"]),
            )
        ],
        **options,
    )
    assert (plus.size, minus.size) == (50, 0)


def test_prepared_skips_zero_weight_components():
    prepared = prepare_inverse_toy_generator(
        model(RealImag(0.0, 0.0)),
        signal_fraction=0.0,
        resolution=32,
        backgrounds=[
            ToyBackground("zero", lambda d: jnp.zeros_like(d["s12"]), fraction=0.0),
            ToyBackground("active", lambda d: jnp.ones_like(d["s12"])),
        ],
    )
    assert prepared.generate(50, seed=1).size == 50


@pytest.mark.parametrize("mode", ["signal", "mixture", "cp"])
def test_high_level_inverse_respects_veto(mode):
    m = model()

    def veto(data):
        return (data["s12"] < 0.8) | (data["s12"] > 1.2)

    options = dict(seed=11, inverse_resolution=32, include_momenta=False)
    if mode == "cp":
        toys = generate_cp_toy(m, m, 1000, plus_veto=veto, minus_veto=veto, **options)
    elif mode == "mixture":
        toys = (
            generate_toy(
                m,
                1000,
                veto=veto,
                signal_fraction=0.5,
                backgrounds=[ToyBackground("bg", lambda d: jnp.ones_like(d["s12"]))],
                **options,
            ),
        )
    else:
        toys = (generate_signal_toy(m, 1000, veto=veto, **options),)
    assert sum(toy.size for toy in toys) == 1000
    assert all(np.all(veto(toy.as_dict())) for toy in toys)
