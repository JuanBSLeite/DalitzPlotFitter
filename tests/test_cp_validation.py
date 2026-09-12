import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    CPBackgroundCategory,
    CPFitSession,
    CPRealImag,
    DecayChannel,
    DecayModel,
    NonResonant,
    Parameter,
    RealImag,
    enable_x64,
)
from dalitzplotfitter.amplitude import AmplitudeComponent, PreparedAmplitudeCache
from dalitzplotfitter.likelihood import CPJointNLL, YieldAsymmetry

enable_x64()


def cache(size=2):
    return PreparedAmplitudeCache.prepare(
        (
            AmplitudeComponent(
                "a", lambda d, p: jnp.ones_like(d["x"]), RealImag(1.0, 0.0)
            ),
        ),
        data={"x": jnp.ones(size)},
        normalization_data={"x": jnp.ones(4)},
        normalization_weights=jnp.ones(4),
        normalize_components=False,
    )


def background(name="b", **kwargs):
    return CPBackgroundCategory(name, jnp.ones(2), jnp.ones(2), 1.0, 1.0, **kwargs)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_extended_rejects_invalid_constants_and_jitted_yields(bad):
    c = cache()
    with pytest.raises(ValueError, match="physical"):
        CPJointNLL(c, c, extended=True, signal_yield=bad)
    nll = CPJointNLL(
        c,
        c,
        extended=True,
        signal_yield=1.0,
        background_categories=(background(yield_=Parameter("nb", 1.0)),),
    )
    value, grad = jax.jit(jax.value_and_grad(lambda nb: nll({"nb": nb})))(bad)
    assert jnp.isposinf(value)
    assert jnp.isfinite(grad)
    assert jnp.isfinite(nll({"nb": 0.0}))


def test_fraction_simplex_and_invalid_signal_fraction():
    c = cache()
    categories = (
        background("a", fraction=Parameter("fa", 0.2)),
        background("b", fraction=Parameter("fb", 0.3)),
        background("c"),
    )
    nll = CPJointNLL(
        c, c, signal_fraction=Parameter("fs", 0.5), background_categories=categories
    )
    evaluate = jax.jit(lambda values: nll(values))
    for values in ({"fa": 0.8, "fb": 0.8}, {"fs": 2.0}, {"fs": -0.1}):
        assert jnp.isposinf(evaluate(values))
    for fs in (0.0, 1.0):
        assert jnp.isfinite(evaluate({"fs": fs, "fa": 0.5, "fb": 0.5}))
    with pytest.raises(ValueError, match="physical"):
        CPJointNLL(c, c, signal_fraction=2.0, background_categories=(background(),))


@pytest.mark.parametrize(
    "bad",
    [jnp.ones((2, 1)), jnp.ones(3), jnp.array([-1.0, 1.0]), jnp.array([jnp.nan, 1.0])],
)
def test_efficiency_rejects_invalid_arrays(bad):
    c = cache()
    with pytest.raises(ValueError):
        CPJointNLL(c, c, plus_efficiency=bad, minus_efficiency=jnp.ones(2))


def test_scalar_efficiency_expands_without_broadcasting():
    c = cache()
    a = CPJointNLL(c, c, plus_efficiency=1.0, minus_efficiency=1.0)
    b = CPJointNLL(c, c, plus_efficiency=jnp.ones(2), minus_efficiency=jnp.ones(2))
    assert a.plus_efficiency.shape == (2,)
    assert a({}) == b({})
    with pytest.raises(ValueError, match="size mismatch"):
        CPJointNLL(
            c, cache(3), signal_fraction=0.5, background_categories=(background(),)
        )


def test_cp_sessions_reuse_normalization_and_fit_charge_asymmetry():
    dx = Parameter.coefficient("dx", 0.05, owner="NR", bounds=(-0.8, 0.8))
    cp = CPRealImag(1.0, 0.0, dx, 0.0)
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    plus, minus = [
        DecayModel(
            channel,
            [NonResonant(cp.for_charge(q))],
            normalization_method="square-dalitz",
            normalization_resolution=12,
        )
        for q in (1, -1)
    ]
    data = plus.generate_phase_space(60, seed=1), minus.generate_phase_space(40, seed=2)
    session = CPFitSession(plus, minus, *data)
    result = session.fit()
    assert result.valid
    expected = (np.sqrt(1.5) - 1) / (np.sqrt(1.5) + 1)
    assert abs(result.values["dx"] - expected) < 1e-5
    assert len(plus._fixed_normalization_templates) == 1
    assert len(minus._fixed_normalization_templates) == 1
    other = CPFitSession(plus, minus, *data)
    assert (
        other.plus_cache.normalization_matrix_fixed
        is session.plus_cache.normalization_matrix_fixed
    )
    weighted = session.with_efficiency(lambda d: 0.5)
    assert jnp.allclose(
        weighted.plus_cache.normalization_matrix_fixed,
        0.5 * session.plus_cache.normalization_matrix_fixed,
    )
    bad = session.with_efficiency(lambda d: jnp.ones((len(d["s12"]), 1)))
    with pytest.raises(ValueError, match="shape"):
        _ = bad.objective
    check = session.minimizer().check_gradient({"dx": 0.05}, print_table=False)
    np.testing.assert_allclose(
        check.jax_gradient, check.finite_difference_gradient, rtol=1e-6, atol=1e-7
    )


def test_cp_session_extended_yield_asymmetry_recovers_raw_counts():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    plus = DecayModel(
        channel, [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz", normalization_resolution=12,
    )
    minus = DecayModel(
        channel, [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz", normalization_resolution=12,
    )
    plus_data = plus.generate_phase_space(700, seed=11)
    minus_data = minus.generate_phase_space(300, seed=12)
    n_s = Parameter("n_s", 900.0, bounds=(1.0, 5000.0))
    a_cp = Parameter("a_cp_yield", 0.0, bounds=(-0.99, 0.99))
    signal_yield = YieldAsymmetry(n_s, a_cp)
    session = CPFitSession(plus, minus, plus_data, minus_data, extended=True, signal_yield=signal_yield)
    result = session.fit()
    assert result.valid
    expected_n_s = 1000.0
    expected_a = (300.0 - 700.0) / 1000.0
    assert abs(result.values["n_s"] - expected_n_s) < 1e-3
    assert abs(result.values["a_cp_yield"] - expected_a) < 1e-6
    check = session.minimizer().check_gradient(session.result_values(result), print_table=False)
    np.testing.assert_allclose(check.jax_gradient, check.finite_difference_gradient, rtol=1e-6, atol=1e-7)


def test_legacy_yield_and_zero_total_density_are_invalid():
    c = cache()
    nll = CPJointNLL(
        c,
        c,
        plus_background=jnp.ones(2),
        minus_background=jnp.ones(2),
        plus_background_normalization=1.0,
        minus_background_normalization=1.0,
        extended=True,
        signal_yield=Parameter("ns", 1.0),
        background_yield=Parameter("nb", 1.0),
    )
    evaluate = jax.jit(lambda ns, nb: nll({"ns": ns, "nb": nb}))
    assert jnp.isposinf(evaluate(1.0, -10.0))
    assert jnp.isposinf(evaluate(0.0, 0.0))
    assert jnp.isfinite(evaluate(0.0, 1.0))
