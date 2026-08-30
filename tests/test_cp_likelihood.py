import jax.numpy as jnp

from dalitzplotfitter import RealImag, enable_x64
from dalitzplotfitter.amplitude import AmplitudeComponent, PreparedAmplitudeCache
from dalitzplotfitter.likelihood import CPJointNLL


enable_x64()


class StaticAmplitude:
    def __init__(self, values):
        self.values = jnp.asarray(values, dtype=jnp.complex128)

    def __call__(self, data, parameters=None):
        n = len(next(iter(data.values())))
        return jnp.resize(self.values, n)


def _cache(amplitude, coefficient, n_data, norm_weights):
    return PreparedAmplitudeCache.prepare(
        (AmplitudeComponent("a", amplitude, coefficient),),
        data={"x": jnp.arange(float(n_data))},
        normalization_data={"x": jnp.arange(float(len(norm_weights)))},
        normalization_weights=jnp.asarray(norm_weights),
        normalize_components=False,
    )


def test_cp_joint_nll_uses_sum_of_charge_integrals():
    plus = _cache(
        StaticAmplitude([1.0 + 0.0j, 2.0 + 0.0j]),
        RealImag(1.0, 0.0),
        3,
        [1.0, 1.0],
    )
    minus = _cache(
        StaticAmplitude([0.5 + 0.0j, 1.0 + 0.0j]),
        RealImag(1.0, 0.0),
        2,
        [1.0, 1.0],
    )
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
    plus = _cache(
        StaticAmplitude([2.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0]
    )
    minus = _cache(
        StaticAmplitude([1.0 + 0.0j]), RealImag(1.0, 0.0), 1, [1.0]
    )
    p_plus, p_minus = CPJointNLL(plus, minus).charge_probabilities({})
    assert jnp.allclose(p_plus, 0.8, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(p_minus, 0.2, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(p_plus + p_minus, 1.0, rtol=1e-12, atol=1e-12)
