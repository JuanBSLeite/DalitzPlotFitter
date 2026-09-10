import inspect

import jax.numpy as jnp
import numpy as np
import pytest
import uproot

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
    prepare_inverse_toy_generator,
)


def _model():
    return DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
    )


def test_inverse_transform_is_default_toy_method():
    assert inspect.signature(generate_signal_toy).parameters["method"].default == "inverse-transform"
    assert inspect.signature(generate_toy).parameters["method"].default == "inverse-transform"
    assert inspect.signature(generate_cp_toy).parameters["method"].default == "inverse-transform"


def test_generate_signal_toy_returns_requested_unweighted_size():
    toy = generate_signal_toy(_model(), 60, seed=10, inverse_resolution=64)
    assert toy.size == 60
    assert jnp.allclose(toy.weights, 1.0)


def test_generate_signal_toy_supports_accept_reject():
    toy = generate_signal_toy(
        _model(),
        60,
        seed=10,
        method="accept-reject",
        pool_size=500,
    )
    assert toy.size == 60
    assert jnp.allclose(toy.weights, 1.0)


def test_generate_signal_toy_supports_inverse_transform():
    toy = generate_signal_toy(
        _model(),
        250,
        seed=101,
        method="inverse-transform",
        inverse_resolution=96,
    )
    assert toy.size == 250
    assert jnp.allclose(toy.weights, 1.0)
    assert np.unique(np.asarray(toy.s12)).size == toy.size
    assert toy.p1 is not None and toy.p2 is not None and toy.p3 is not None


def test_inverse_transform_can_skip_four_momenta_without_changing_invariants():
    model = _model()
    full = generate_signal_toy(
        model,
        500,
        seed=102,
        inverse_resolution=96,
        include_momenta=True,
    )
    compact = generate_signal_toy(
        model,
        500,
        seed=102,
        inverse_resolution=96,
        include_momenta=False,
    )

    assert compact.p1 is None and compact.p2 is None and compact.p3 is None
    assert jnp.array_equal(compact.s12, full.s12)
    assert jnp.array_equal(compact.s13, full.s13)
    assert jnp.array_equal(compact.s23, full.s23)
    assert jnp.array_equal(compact.weights, full.weights)
    assert compact.nbytes * 4 == full.nbytes


def test_accept_reject_can_return_compact_toy():
    toy = generate_signal_toy(
        _model(),
        100,
        seed=103,
        method="accept-reject",
        pool_size=500,
        include_momenta=False,
    )
    assert toy.p1 is None and toy.p2 is None and toy.p3 is None
    assert toy.size == 100


def test_prepared_inverse_generator_can_return_compact_toy():
    prepared = prepare_inverse_toy_generator(_model(), resolution=80)
    toy = prepared.generate(120, seed=104, include_momenta=False)
    assert toy.size == 120
    assert toy.p1 is None and toy.p2 is None and toy.p3 is None


def test_resample_is_not_a_public_toy_method():
    with pytest.raises(ValueError, match="method must be one of"):
        generate_toy(_model(), 20, method="resample")


def test_generate_toy_rejects_unknown_sampling_method():
    with pytest.raises(ValueError, match="method must be one of"):
        generate_toy(_model(), 20, method="inverse")


def test_inverse_transform_rejects_accept_reject_only_options():
    with pytest.raises(ValueError, match="pool_size applies only"):
        generate_toy(
            _model(),
            20,
            method="inverse-transform",
            pool_size=100,
            inverse_resolution=64,
        )


def test_generate_toy_supports_signal_background_mixture():
    background = ToyBackground("comb", lambda d: jnp.ones_like(d["s12"]))
    toy = generate_toy(
        _model(),
        80,
        signal_fraction=0.7,
        backgrounds=(background,),
        seed=11,
        method="accept-reject",
        pool_size=600,
    )
    assert toy.size == 80
    assert jnp.allclose(toy.weights, 1.0)


def test_generate_toy_inverse_supports_signal_background_mixture():
    background = ToyBackground("comb", lambda d: jnp.ones_like(d["s12"]))
    toy = generate_toy(
        _model(),
        120,
        signal_fraction=0.7,
        backgrounds=(background,),
        seed=111,
        inverse_resolution=80,
    )
    assert toy.size == 120
    assert jnp.allclose(toy.weights, 1.0)


def test_prepared_inverse_generator_can_be_reused():
    prepared = prepare_inverse_toy_generator(_model(), resolution=80)
    first = prepared.generate(100, seed=1)
    second = prepared.generate(100, seed=2)
    assert first.size == 100
    assert second.size == 100
    assert not jnp.allclose(first.s12, second.s12)


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
        inverse_resolution=80,
    )
    assert plus_toy.size + minus_toy.size == 400
    assert plus_toy.size > minus_toy.size
    assert jnp.allclose(plus_toy.weights, 1.0)
    assert jnp.allclose(minus_toy.weights, 1.0)


def test_generate_cp_toy_inverse_preserves_total_event_count_and_charge_model():
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
        seed=112,
        method="inverse-transform",
        inverse_resolution=80,
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
        method="accept-reject",
        pool_size=500,
    )
    assert plus_toy.size + minus_toy.size == 120


def test_generate_cp_toy_can_write_one_root_tree_with_charge(tmp_path):
    cp = CPRealImag(1.0, 0.0, 0.0, 0.0)
    plus = DecayModel(
        DecayChannel("B+", ("K+", "pi+", "pi-")),
        [NonResonant(cp.for_charge(+1))],
        normalization_method="square-dalitz",
        normalization_resolution=8,
        normalization_pair=(0, 2),
    )
    minus = DecayModel(
        DecayChannel("B-", ("K-", "pi-", "pi+")),
        [NonResonant(cp.for_charge(-1))],
        normalization_method="square-dalitz",
        normalization_resolution=8,
        normalization_pair=(0, 2),
    )
    path = tmp_path / "cp_toy.root"
    plus_toy, minus_toy = generate_cp_toy(
        plus,
        minus,
        80,
        seed=42,
        inverse_resolution=64,
        output_root=path,
    )
    with uproot.open(path) as root_file:
        tree = root_file["DecayTree"]
        arrays = tree.arrays(["charge"], library="np")
        charge = arrays["charge"]
        assert tree.num_entries == 80
        assert np.count_nonzero(charge == 1) == plus_toy.size
        assert np.count_nonzero(charge == -1) == minus_toy.size
