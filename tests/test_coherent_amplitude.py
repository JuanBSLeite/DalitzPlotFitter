import jax.numpy as jnp

from dalitzplotfitter import RealImag
from dalitzplotfitter.amplitude import (
    AmplitudeComponent,
    CoherentAmplitudeModel,
    ConstantAmplitude,
)


def test_coherent_sum_contains_interference():
    data = {"x": jnp.ones(16)}
    first = AmplitudeComponent("a", ConstantAmplitude(), RealImag(1.0, 0.0))
    second = AmplitudeComponent("b", ConstantAmplitude(), RealImag(0.5, 0.0))
    intensity = CoherentAmplitudeModel((first, second)).intensity(data)
    assert jnp.allclose(intensity, 1.5**2)
    assert not jnp.allclose(intensity, 1.0**2 + 0.5**2)


def test_imaginary_coefficient_changes_interference():
    data = {"x": jnp.ones(8)}
    reference = AmplitudeComponent("a", ConstantAmplitude(), RealImag(1.0, 0.0))
    constructive = AmplitudeComponent("b", ConstantAmplitude(), RealImag(1.0, 0.0))
    quadrature = AmplitudeComponent("b", ConstantAmplitude(), RealImag(0.0, 1.0))
    high = CoherentAmplitudeModel((reference, constructive)).intensity(data)
    orthogonal = CoherentAmplitudeModel((reference, quadrature)).intensity(data)
    assert jnp.allclose(high, 4.0)
    assert jnp.allclose(orthogonal, 2.0)
