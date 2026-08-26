import jax.numpy as jnp

from dalitzplotfitter.coefficients import (
    CartesianCP,
    FitCartesian,
    Flavor,
    MagPhase,
    RealImag,
)
from dalitzplotfitter.fit import Parameter


def test_cp_conserving_sets_ignore_flavor():
    for coefficient in (MagPhase(2.0, 0.3), RealImag(1.0, -0.4)):
        assert jnp.allclose(
            coefficient.value(Flavor.PARTICLE),
            coefficient.value(Flavor.ANTIPARTICLE),
        )


def test_fit_cartesian_is_x_plus_iy_and_ignores_flavor():
    x = Parameter.coefficient("x", 0.3)
    y = Parameter.coefficient("y", -0.4)
    coefficient = FitCartesian(x, y)
    values = {"x": 0.7, "y": -0.2}
    assert jnp.allclose(coefficient.value(values=values), 0.7 - 0.2j)
    assert jnp.allclose(
        coefficient.value(Flavor.PARTICLE, values),
        coefficient.value(Flavor.ANTIPARTICLE, values),
    )


def test_cartesian_cp_sign_convention():
    coefficient = CartesianCP(x=1.0, y=2.0, dx=0.1, dy=0.2)
    assert jnp.allclose(coefficient.value(Flavor.PARTICLE), 1.1 + 2.2j)
    assert jnp.allclose(coefficient.value(Flavor.ANTIPARTICLE), 0.9 + 1.8j)
