import jax.numpy as jnp

from dalitzplotfitter import (
    CPToyBackground,
    CPRealImag,
    DecayChannel,
    DecayModel,
    NonResonant,
    Parameter,
    RealImag,
    ToyBackground,
    generate_cp_toy,
    generate_signal_toy,
    generate_toy,
)


def _model():
    return DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
    )


def test_generate_signal_toy_returns_requested_unweighted_size():
    toy = generate_signal_toy(_model(), 60, seed=10, pool_size=500)
    assert toy.size == 60
    assert jnp.allclose(toy.weights, 1.0)


def test_generate_toy_supports_signal_background_mixture():
    background = ToyBackground("comb", lambda d: jnp.ones_like(d["s12"]))
    toy = generate_toy(
        _model(),
        80,
        signal_fraction=0.7,
        backgrounds=(background,),
        seed=11,
        pool_size=600,
    )
    assert toy.size == 80
    assert jnp.allclose(toy.weights, 1.0)


def test_generate_cp_toy_preserves_total_event_count_and_charge_model():
    x = Parameter.coefficient("NR.x", 1.0, owner="NR")
    dx = Parameter.coefficient("NR.dx", 0.25, owner="NR")
    cp = CPRealImag(x, 0.0, dx, 0.0)
    plus = DecayModel(
        DecayChannel("B+", ("K+", "pi+", "pi-")),
        [NonResonant(cp.for_charge(+1))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
        normalization_pair=(0, 2),
    )
    minus = DecayModel(
        DecayChannel("B-", ("K-", "pi-", "pi+")),
        [NonResonant(cp.for_charge(-1))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
        normalization_pair=(0, 2),
    )
    plus_toy, minus_toy = generate_cp_toy(
        plus,
        minus,
        400,
        parameters={"NR.x": 1.0, "NR.dx": 0.25},
        seed=12,
        pool_size=800,
    )
    assert plus_toy.size + minus_toy.size == 400
    assert plus_toy.size > minus_toy.size
    assert jnp.allclose(plus_toy.weights, 1.0)
    assert jnp.allclose(minus_toy.weights, 1.0)


def test_generate_cp_toy_supports_backgrounds():
    cp = CPRealImag(1.0, 0.0, 0.0, 0.0)
    plus = DecayModel(
        DecayChannel("B+", ("K+", "pi+", "pi-")),
        [NonResonant(cp.for_charge(+1))],
        normalization_method="square-dalitz",
        normalization_resolution=10,
        normalization_pair=(0, 2),
    )
    minus = DecayModel(
        DecayChannel("B-", ("K-", "pi-", "pi+")),
        [NonResonant(cp.for_charge(-1))],
        normalization_method="square-dalitz",
        normalization_resolution=10,
        normalization_pair=(0, 2),
    )
    background = CPToyBackground("comb", lambda d: jnp.ones_like(d["s12"]))
    plus_toy, minus_toy = generate_cp_toy(
        plus,
        minus,
        120,
        signal_fraction=0.75,
        backgrounds=(background,),
        seed=13,
        pool_size=500,
    )
    assert plus_toy.size + minus_toy.size == 120
