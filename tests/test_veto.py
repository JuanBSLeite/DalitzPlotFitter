import jax.numpy as jnp

from dalitzplotfitter import CompositeVeto, FunctionalVeto, MassWindowVeto, SignalPDF
from dalitzplotfitter.integration import GridIntegrator
from dalitzplotfitter.kinematics import PhaseSpaceSample


def _sample():
    return PhaseSpaceSample(
        s12=jnp.asarray([0.25, 1.00, 2.25, 4.00]),
        s13=jnp.asarray([0.36, 1.21, 2.56, 4.41]),
        s23=jnp.asarray([0.49, 1.44, 2.89, 4.84]),
        weights=jnp.ones(4),
    )


def test_mass_window_veto_uses_mass_not_mass_squared():
    veto = MassWindowVeto((0, 1), 0.9, 1.1)
    mask = veto(_sample().as_dict())
    assert jnp.array_equal(mask, jnp.asarray([True, False, True, True]))


def test_composite_veto_and_sample_filtering():
    mass_veto = MassWindowVeto((0, 1), 0.9, 1.1)
    extra = FunctionalVeto(lambda data: data["s13"] < 4.0)
    veto = CompositeVeto(mass_veto, extra)
    filtered = veto.apply(_sample())
    assert filtered.size == 2
    assert jnp.array_equal(filtered.s12, jnp.asarray([0.25, 2.25]))


def test_signal_pdf_veto_enters_numerator_and_normalization():
    sample = _sample()
    veto = MassWindowVeto((0, 1), 0.9, 1.1)
    pdf = SignalPDF(
        intensity=lambda data, parameters: jnp.ones_like(data["s12"]),
        integrator=GridIntegrator(sample),
        veto=veto,
    )
    normalization = pdf.normalization({})
    assert jnp.allclose(normalization, 0.75)
    values = pdf(sample.as_dict(), {})
    assert values[1] <= 1e-299
    assert jnp.allclose(values[jnp.asarray([0, 2, 3])], 1.0 / 0.75)
