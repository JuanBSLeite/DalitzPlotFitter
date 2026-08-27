import jax.numpy as jnp

from dalitzplotfitter.coefficients import RealImag
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
