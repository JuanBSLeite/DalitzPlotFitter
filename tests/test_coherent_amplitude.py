import jax.numpy as jnp

from dalitzplotfitter import MagPhase
from dalitzplotfitter.amplitude import (
    AmplitudeComponent,
    CoherentAmplitudeModel,
    ConstantAmplitude,
)


def test_coherent_sum_contains_interference():
    data = {"x": jnp.ones(16)}
    first = AmplitudeComponent(
        "a",
        ConstantAmplitude(1.0 + 0.0j),
        MagPhase(r=1.0, phi=0.0),
    )
    second = AmplitudeComponent(
        "b",
        ConstantAmplitude(1.0 + 0.0j),
        MagPhase(r=0.5, phi=0.0),
    )
    model = CoherentAmplitudeModel((first, second))

    intensity = model.intensity(data)
    incoherent = 1.0**2 + 0.5**2

    assert jnp.allclose(intensity, 1.5**2)
    assert not jnp.allclose(intensity, incoherent)


def test_phase_changes_interference():
    data = {"x": jnp.ones(8)}
    first = AmplitudeComponent("a", ConstantAmplitude(), MagPhase(r=1.0, phi=0.0))
    constructive = AmplitudeComponent(
        "b", ConstantAmplitude(), MagPhase(r=1.0, phi=0.0)
    )
    destructive = AmplitudeComponent(
        "b", ConstantAmplitude(), MagPhase(r=1.0, phi=jnp.pi)
    )

    high = CoherentAmplitudeModel((first, constructive)).intensity(data)
    low = CoherentAmplitudeModel((first, destructive)).intensity(data)

    assert jnp.allclose(high, 4.0)
    assert jnp.allclose(low, 0.0, atol=1e-12)
