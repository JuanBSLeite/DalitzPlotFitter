import jax.numpy as jnp

from dalitzplotfitter import BackgroundCategory, MultiBackgroundNLL, Parameter


def test_nonextended_multiple_background_categories():
    signal = jnp.asarray([0.2, 0.8])
    comb = BackgroundCategory(
        "combinatorial",
        values=jnp.asarray([1.0, 3.0]),
        normalization=2.0,
        fraction=Parameter("f_comb", 0.25),
    )
    partial = BackgroundCategory(
        "partially_reconstructed",
        values=jnp.asarray([3.0, 1.0]),
        normalization=2.0,
    )
    nll = MultiBackgroundNLL(
        signal_density=lambda values: signal,
        backgrounds=(comb, partial),
        signal_fraction=Parameter("f_sig", 0.60),
    )
    density = nll.density({"f_sig": 0.60, "f_comb": 0.25})
    b1 = comb.density
    b2 = partial.density
    expected = 0.60 * signal + 0.40 * (0.25 * b1 + 0.75 * b2)
    assert jnp.allclose(density, expected)
    assert jnp.allclose(nll.background_weights({"f_comb": 0.25}), jnp.asarray([0.25, 0.75]))


def test_extended_multiple_background_yields():
    signal = jnp.asarray([0.2, 0.8])
    comb = BackgroundCategory(
        "combinatorial",
        values=jnp.asarray([1.0, 3.0]),
        normalization=2.0,
        yield_=Parameter("n_comb", 20.0),
    )
    misid = BackgroundCategory(
        "misid",
        values=jnp.asarray([2.0, 2.0]),
        normalization=2.0,
        yield_=Parameter("n_misid", 10.0),
    )
    nll = MultiBackgroundNLL(
        signal_density=lambda values: signal,
        backgrounds=(comb, misid),
        extended=True,
        signal_yield=Parameter("n_sig", 70.0),
    )
    values = {"n_sig": 70.0, "n_comb": 20.0, "n_misid": 10.0}
    expected_density = 70.0 * signal + 20.0 * comb.density + 10.0 * misid.density
    assert jnp.allclose(nll.density(values), expected_density)
    assert jnp.allclose(nll.expected_events(values), 100.0)
    expected_nll = 100.0 - jnp.sum(jnp.log(expected_density))
    assert jnp.allclose(nll(values), expected_nll)


def test_background_category_names_must_be_unique():
    a = BackgroundCategory("same", jnp.ones(2), 1.0)
    b = BackgroundCategory("same", jnp.ones(2), 1.0)
    try:
        MultiBackgroundNLL(lambda values: jnp.ones(2), (a, b), signal_fraction=0.5)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate background names should fail")
