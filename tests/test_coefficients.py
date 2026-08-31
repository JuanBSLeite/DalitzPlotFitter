import jax.numpy as jnp

from dalitzplotfitter.coefficients import CPRealImag, RealImag
from dalitzplotfitter.fit import Parameter


def test_real_imag_is_x_plus_iy():
    coefficient = RealImag(1.2, -0.4)
    assert jnp.allclose(coefficient.value(), 1.2 - 0.4j)


def test_real_imag_resolves_fit_parameters():
    x = Parameter.coefficient("x", 0.3)
    y = Parameter.coefficient("y", -0.4)
    coefficient = RealImag(x, y)
    assert coefficient.parameters == (x, y)
    assert jnp.allclose(
        coefficient.value({"x": 0.7, "y": -0.2}),
        0.7 - 0.2j,
    )


def test_cp_real_imag_builds_charge_conjugate_coefficients():
    plus = CPRealImag(0.8, -0.2, 0.1, 0.05, charge=+1)
    minus = plus.for_charge(-1)
    assert jnp.allclose(plus.value(), 0.9 - 0.15j)
    assert jnp.allclose(minus.value(), 0.7 - 0.25j)


def test_cp_real_imag_resolves_shared_fit_parameters():
    x = Parameter.coefficient("x", 0.8, owner="signal")
    y = Parameter.coefficient("y", -0.2, owner="signal")
    dx = Parameter.coefficient("dx", 0.1, owner="signal")
    dy = Parameter.coefficient("dy", 0.05, owner="signal")
    plus = CPRealImag(x, y, dx, dy, charge=+1)
    minus = plus.for_charge(-1)
    values = {"x": 1.0, "y": 0.3, "dx": -0.2, "dy": 0.1}
    assert plus.parameters == (x, y, dx, dy)
    assert minus.parameters == (x, y, dx, dy)
    assert jnp.allclose(plus.value(values), 0.8 + 0.4j)
    assert jnp.allclose(minus.value(values), 1.2 + 0.2j)


def test_cp_real_imag_rejects_nonphysical_charge_label():
    try:
        CPRealImag(1.0, 0.0, charge=0)
    except ValueError as exc:
        assert "charge" in str(exc)
    else:
        raise AssertionError("CPRealImag accepted charge=0")
