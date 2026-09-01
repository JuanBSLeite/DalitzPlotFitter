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
    expected = (
        -jnp.sum(jnp.log(ip))
        -jnp.sum(jnp.log(im))
        + (ip.shape[0] + im.shape[0]) * jnp.log(np_ + nm_)
    )
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
    expected = -jnp.sum(jnp.log(plus_eff_data * ip / total)) - jnp.sum(
        jnp.log(minus_eff_data * im / total)
    )
    assert jnp.allclose(nll({}), expected, rtol=1e-12, atol=1e-12)


def test_cp_joint_nll_supports_joint_background_mixture():
    plus = _cache(StaticAmplitude([1.0 + 0.0j]), RealImag(1.0, 0.0), 2, [1.0])
    minus = _cache(StaticAmplitude([2.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0])
    fraction = Parameter("background_fraction", 0.25)
    nll = CPJointNLL(
        plus,
        minus,
        plus_background=jnp.asarray([2.0, 1.0]),
        minus_background=jnp.asarray([3.0]),
        plus_background_normalization=2.0,
        minus_background_normalization=1.0,
        background_fraction=fraction,
    )

    sp, sm, _, _ = nll._signal_densities({"background_fraction": 0.25})
    bp, bm = nll._background_densities()
    expected_plus = 0.75 * sp + 0.25 * bp
    expected_minus = 0.75 * sm + 0.25 * bm
    pdf_plus, pdf_minus = nll.densities({"background_fraction": 0.25})
    assert jnp.allclose(pdf_plus, expected_plus, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(pdf_minus, expected_minus, rtol=1e-12, atol=1e-12)

    p_plus, p_minus = nll.charge_probabilities({"background_fraction": 0.25})
    signal_plus = plus.normalization({}) / (plus.normalization({}) + minus.normalization({}))
    background_plus = 2.0 / 3.0
    assert jnp.allclose(p_plus, 0.75 * signal_plus + 0.25 * background_plus)
    assert jnp.allclose(p_plus + p_minus, 1.0, rtol=1e-12, atol=1e-12)


def test_cp_joint_nll_rejects_partial_background_configuration():
    plus = _cache(StaticAmplitude([1.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0])
    minus = _cache(StaticAmplitude([1.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0])
    try:
        CPJointNLL(plus, minus, plus_background=jnp.ones(1))
    except ValueError as exc:
        assert "background mixture requires" in str(exc)
    else:
        raise AssertionError("partial background configuration should fail")
