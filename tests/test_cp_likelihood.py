import jax.numpy as jnp

from dalitzplotfitter import RealImag, enable_x64
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


def _cache(amplitude, coefficient, n_data, norm_weights, efficiency_normalization=None):
    return PreparedAmplitudeCache.prepare(
        (AmplitudeComponent("a", amplitude, coefficient),),
        data={"x": jnp.arange(float(n_data))},
        normalization_data={"x": jnp.arange(float(len(norm_weights)))},
        normalization_weights=jnp.asarray(norm_weights),
        efficiency_normalization=efficiency_normalization,
        normalize_components=False,
    )


def test_cp_joint_nll_uses_sum_of_charge_integrals():
    plus = _cache(StaticAmplitude([1.0 + 0.0j, 2.0 + 0.0j]), RealImag(1.0, 0.0), 3, [1.0, 1.0])
    minus = _cache(StaticAmplitude([0.5 + 0.0j, 1.0 + 0.0j]), RealImag(1.0, 0.0), 2, [1.0, 1.0])
    nll = CPJointNLL(plus, minus)
    ip, np_ = plus.evaluate({})
    im, nm_ = minus.evaluate({})
    expected = -jnp.sum(jnp.log(ip)) - jnp.sum(jnp.log(im)) + (ip.shape[0] + im.shape[0]) * jnp.log(np_ + nm_)
    assert jnp.allclose(nll({}), expected, rtol=1e-12, atol=1e-12)


def test_cp_joint_charge_probabilities_follow_integrated_rates():
    plus = _cache(StaticAmplitude([2.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0])
    minus = _cache(StaticAmplitude([1.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0])
    p_plus, p_minus = CPJointNLL(plus, minus).charge_probabilities({})
    assert jnp.allclose(p_plus, 0.8, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(p_minus, 0.2, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(p_plus + p_minus, 1.0, rtol=1e-12, atol=1e-12)


def test_cp_joint_nll_supports_efficiency_weighted_signal():
    plus_eff_norm = jnp.asarray([0.5, 1.0])
    minus_eff_norm = jnp.asarray([1.0, 0.5])
    plus = _cache(StaticAmplitude([1.0 + 0.0j, 2.0 + 0.0j]), RealImag(1.0, 0.0), 2, [1.0, 1.0], plus_eff_norm)
    minus = _cache(StaticAmplitude([0.5 + 0.0j, 1.0 + 0.0j]), RealImag(1.0, 0.0), 2, [1.0, 1.0], minus_eff_norm)
    plus_eff_data = jnp.asarray([0.8, 0.6])
    minus_eff_data = jnp.asarray([0.7, 0.9])
    nll = CPJointNLL(plus, minus, plus_efficiency=plus_eff_data, minus_efficiency=minus_eff_data)
    ip, np_ = plus.evaluate({})
    im, nm_ = minus.evaluate({})
    total = np_ + nm_
    expected = -jnp.sum(jnp.log(plus_eff_data * ip / total)) - jnp.sum(jnp.log(minus_eff_data * im / total))
    assert jnp.allclose(nll({}), expected, rtol=1e-12, atol=1e-12)


def test_cp_joint_nll_uses_signal_fraction_for_background_mixture():
    plus = _cache(StaticAmplitude([1.0 + 0.0j]), RealImag(1.0, 0.0), 2, [1.0])
    minus = _cache(StaticAmplitude([2.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0])
    fraction = Parameter("signal_fraction", 0.75)
    nll = CPJointNLL(
        plus,
        minus,
        plus_background=jnp.asarray([2.0, 1.0]),
        minus_background=jnp.asarray([3.0]),
        plus_background_normalization=2.0,
        minus_background_normalization=1.0,
        signal_fraction=fraction,
    )
    sp, sm, _, _ = nll._signal_densities({"signal_fraction": 0.75})
    bp, bm = nll._background_densities()
    pdf_plus, pdf_minus = nll.densities({"signal_fraction": 0.75})
    assert jnp.allclose(pdf_plus, 0.75 * sp + 0.25 * bp, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(pdf_minus, 0.75 * sm + 0.25 * bm, rtol=1e-12, atol=1e-12)
    p_plus, p_minus = nll.charge_probabilities({"signal_fraction": 0.75})
    signal_plus = plus.normalization({}) / (plus.normalization({}) + minus.normalization({}))
    assert jnp.allclose(p_plus, 0.75 * signal_plus + 0.25 * (2.0 / 3.0))
    assert jnp.allclose(p_plus + p_minus, 1.0, rtol=1e-12, atol=1e-12)


def test_cp_joint_extended_signal_only_uses_poisson_term():
    plus = _cache(StaticAmplitude([1.0 + 0.0j]), RealImag(1.0, 0.0), 2, [1.0])
    minus = _cache(StaticAmplitude([2.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0])
    signal_yield = Parameter("signal_yield", 3.0)
    nll = CPJointNLL(plus, minus, extended=True, signal_yield=signal_yield)
    sp, sm = CPJointNLL(plus, minus).densities({})
    values = {"signal_yield": 3.0}
    expected = 3.0 - jnp.sum(jnp.log(3.0 * sp)) - jnp.sum(jnp.log(3.0 * sm))
    assert jnp.allclose(nll(values), expected, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(nll.expected_events(values), 3.0)


def test_cp_joint_extended_signal_background_uses_yields():
    plus = _cache(StaticAmplitude([1.0 + 0.0j]), RealImag(1.0, 0.0), 2, [1.0])
    minus = _cache(StaticAmplitude([2.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0])
    n_sig = Parameter("signal_yield", 90.0)
    n_bkg = Parameter("background_yield", 10.0)
    nll = CPJointNLL(
        plus,
        minus,
        plus_background=jnp.asarray([2.0, 1.0]),
        minus_background=jnp.asarray([3.0]),
        plus_background_normalization=2.0,
        minus_background_normalization=1.0,
        extended=True,
        signal_yield=n_sig,
        background_yield=n_bkg,
    )
    values = {"signal_yield": 90.0, "background_yield": 10.0}
    (sp, sm), background = nll.component_densities(values)
    bp, bm = background
    ep, em = nll.densities(values)
    assert jnp.allclose(ep, 90.0 * sp + 10.0 * bp)
    assert jnp.allclose(em, 90.0 * sm + 10.0 * bm)
    expected = 100.0 - jnp.sum(jnp.log(ep)) - jnp.sum(jnp.log(em))
    assert jnp.allclose(nll(values), expected, rtol=1e-12, atol=1e-12)
    p_plus, p_minus = nll.charge_probabilities(values)
    signal_plus = plus.normalization({}) / (plus.normalization({}) + minus.normalization({}))
    expected_plus = (90.0 * signal_plus + 10.0 * (2.0 / 3.0)) / 100.0
    assert jnp.allclose(p_plus, expected_plus)
    assert jnp.allclose(p_plus + p_minus, 1.0, rtol=1e-12, atol=1e-12)


def test_cp_joint_nll_rejects_inconsistent_modes():
    plus = _cache(StaticAmplitude([1.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0])
    minus = _cache(StaticAmplitude([1.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0])
    try:
        CPJointNLL(plus, minus, plus_background=jnp.ones(1))
    except ValueError as exc:
        assert "background requires" in str(exc)
    else:
        raise AssertionError("partial background configuration should fail")

    try:
        CPJointNLL(plus, minus, extended=True)
    except ValueError as exc:
        assert "signal_yield" in str(exc)
    else:
        raise AssertionError("extended mode without signal_yield should fail")
