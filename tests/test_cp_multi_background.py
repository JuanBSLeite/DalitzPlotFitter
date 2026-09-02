import jax.numpy as jnp

from dalitzplotfitter import CPBackgroundCategory, RealImag, enable_x64
from dalitzplotfitter.amplitude import AmplitudeComponent, PreparedAmplitudeCache
from dalitzplotfitter.fit import Parameter
from dalitzplotfitter.likelihood import CPJointNLL


enable_x64()


class StaticAmplitude:
    def __init__(self, values):
        self.values = jnp.asarray(values, dtype=jnp.complex128)

    def __call__(self, data, parameters=None):
        n = len(next(iter(data.values())))
        return jnp.resize(self.values, n)


def _cache(value, n_data):
    return PreparedAmplitudeCache.prepare(
        (AmplitudeComponent("a", StaticAmplitude([value]), RealImag(1.0, 0.0)),),
        data={"x": jnp.arange(float(n_data))},
        normalization_data={"x": jnp.arange(1.0)},
        normalization_weights=jnp.ones(1),
        normalize_components=False,
    )


def test_cp_joint_supports_multiple_background_categories():
    plus = _cache(2.0 + 0.0j, 2)
    minus = _cache(1.0 + 0.0j, 1)
    comb = CPBackgroundCategory(
        "comb",
        plus_values=jnp.asarray([2.0, 2.0]),
        minus_values=jnp.asarray([1.0]),
        plus_normalization=2.0,
        minus_normalization=1.0,
        fraction=Parameter("f_comb", 0.25),
    )
    misid = CPBackgroundCategory(
        "misid",
        plus_values=jnp.asarray([1.0, 3.0]),
        minus_values=jnp.asarray([2.0]),
        plus_normalization=3.0,
        minus_normalization=2.0,
    )
    nll = CPJointNLL(
        plus,
        minus,
        background_categories=(comb, misid),
        signal_fraction=Parameter("f_sig", 0.60),
    )
    values = {"f_sig": 0.60, "f_comb": 0.25}
    sp, sm, _, _ = nll._signal_densities(values)
    expected_plus = 0.60 * sp + 0.40 * (
        0.25 * comb.plus_density + 0.75 * misid.plus_density
    )
    expected_minus = 0.60 * sm + 0.40 * (
        0.25 * comb.minus_density + 0.75 * misid.minus_density
    )
    dp, dm = nll.densities(values)
    assert jnp.allclose(dp, expected_plus)
    assert jnp.allclose(dm, expected_minus)
    pplus, pminus = nll.charge_probabilities(values)
    assert jnp.allclose(pplus + pminus, 1.0)


def test_cp_joint_extended_uses_independent_background_yields():
    plus = _cache(1.0 + 0.0j, 1)
    minus = _cache(1.0 + 0.0j, 1)
    comb = CPBackgroundCategory(
        "comb",
        plus_values=jnp.ones(1),
        minus_values=jnp.ones(1),
        plus_normalization=1.0,
        minus_normalization=1.0,
        yield_=Parameter("n_comb", 20.0),
    )
    misid = CPBackgroundCategory(
        "misid",
        plus_values=jnp.ones(1),
        minus_values=jnp.ones(1),
        plus_normalization=1.0,
        minus_normalization=1.0,
        yield_=Parameter("n_misid", 10.0),
    )
    nll = CPJointNLL(
        plus,
        minus,
        background_categories=(comb, misid),
        extended=True,
        signal_yield=Parameter("n_sig", 70.0),
    )
    values = {"n_sig": 70.0, "n_comb": 20.0, "n_misid": 10.0}
    assert jnp.allclose(nll.expected_events(values), 100.0)
