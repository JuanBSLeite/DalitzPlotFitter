import jax.numpy as jnp

from dalitzplotfitter.coefficients import (
    CartesianCP,
    Flavor,
    MagPhase,
    RealImag,
)


def test_cp_conserving_sets_ignore_flavor():
    for coefficient in (MagPhase(2.0, 0.3), RealImag(1.0, -0.4)):
        assert jnp.allclose(
            coefficient.value(Flavor.PARTICLE),
            coefficient.value(Flavor.ANTIPARTICLE),
        )


def test_cartesian_cp_sign_convention():
    coefficient = CartesianCP(x=1.0, y=2.0, dx=0.1, dy=0.2)
    assert jnp.allclose(coefficient.value(Flavor.PARTICLE), 1.1 + 2.2j)
    assert jnp.allclose(coefficient.value(Flavor.ANTIPARTICLE), 0.9 + 1.8j)
