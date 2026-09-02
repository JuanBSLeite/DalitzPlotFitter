import jax.numpy as jnp

from dalitzplotfitter import (
    ConvolvedPDF1D,
    FactorizedDensity,
    Gaussian1D,
    GaussianResolution1D,
    Parameter,
)


def _trapz(y, x):
    return jnp.trapezoid(y, x)


def test_convolved_pdf_is_normalized_on_finite_observed_window():
    truth = Gaussian1D(mean=0.1, sigma=0.55, low=-2.0, high=2.0)
    resolution = GaussianResolution1D(sigma=0.35)
    pdf = ConvolvedPDF1D(
        truth,
        resolution,
        true_low=-2.0,
        true_high=2.0,
        observed_low=-1.2,
        observed_high=1.5,
        order=128,
    )
    x = jnp.linspace(-1.2, 1.5, 12001)
    integral = _trapz(pdf(x), x)
    assert jnp.isclose(integral, 1.0, rtol=4e-4, atol=4e-4)
    assert pdf.normalization() < 1.0


def test_narrow_resolution_approaches_original_pdf_away_from_boundaries():
    truth = Gaussian1D(mean=0.15, sigma=0.5, low=-3.0, high=3.0)
    pdf = ConvolvedPDF1D(
        truth,
        GaussianResolution1D(sigma=0.01),
        true_low=-3.0,
        true_high=3.0,
        observed_low=-3.0,
        observed_high=3.0,
        order=256,
    )
    x = jnp.asarray([-0.8, -0.2, 0.15, 0.7, 1.1])
    assert jnp.allclose(pdf(x), truth(x), rtol=3e-3, atol=3e-4)


def test_resolution_bias_moves_reconstructed_mean():
    truth = Gaussian1D(mean=0.0, sigma=0.35, low=-3.0, high=3.0)
    pdf = ConvolvedPDF1D(
        truth,
        GaussianResolution1D(sigma=0.2, bias=0.4),
        true_low=-3.0,
        true_high=3.0,
        observed_low=-3.0,
        observed_high=3.0,
        order=160,
    )
    x = jnp.linspace(-3.0, 3.0, 16001)
    values = pdf(x)
    mean = _trapz(x * values, x)
    assert jnp.isclose(mean, 0.4, atol=2e-3)


def test_convolution_parameters_are_resolved_at_evaluation_time():
    sigma = Parameter("resolution.sigma", 0.25, bounds=(0.01, 1.0))
    truth = Gaussian1D(mean=0.0, sigma=0.4, low=-2.5, high=2.5)
    pdf = ConvolvedPDF1D(
        truth,
        GaussianResolution1D(sigma=sigma),
        true_low=-2.5,
        true_high=2.5,
        observed_low=-2.5,
        observed_high=2.5,
        order=128,
    )
    x = jnp.asarray([0.0, 0.7])
    narrow = pdf(x, {"resolution.sigma": 0.1})
    broad = pdf(x, {"resolution.sigma": 0.6})
    assert narrow[0] > broad[0]
    assert narrow[1] < broad[1]


def test_convolved_pdf_works_inside_factorized_density():
    mass_pdf = ConvolvedPDF1D(
        Gaussian1D(mean=5.28, sigma=0.015, low=5.20, high=5.36),
        GaussianResolution1D(sigma=0.010),
        true_low=5.20,
        true_high=5.36,
        observed_low=5.20,
        observed_high=5.36,
        order=96,
    )
    base = lambda pars: jnp.asarray([0.25, 0.75])
    masses = jnp.asarray([5.28, 5.31])
    density = FactorizedDensity(
        base_density=base,
        observables={"mass": masses},
        pdfs={"mass": mass_pdf},
    )
    assert jnp.allclose(density({}), base({}) * mass_pdf(masses, {}))
