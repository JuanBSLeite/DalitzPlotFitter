import jax.numpy as jnp

from dalitzplotfitter import (
    BreitWigner1D,
    Exponential1D,
    FactorizedDensity,
    Gaussian1D,
    Histogram1D,
    LineshapeIntensity1D,
    RelativisticBreitWigner,
    ResonanceContext,
)


def _trapz(y, x):
    return jnp.trapezoid(y, x)


def test_gaussian_is_normalized_on_finite_interval():
    pdf = Gaussian1D(mean=0.2, sigma=0.7, low=-2.0, high=2.5)
    x = jnp.linspace(-2.0, 2.5, 20001)
    assert jnp.isclose(_trapz(pdf(x), x), 1.0, rtol=2e-4, atol=2e-4)


def test_breit_wigner_is_normalized_on_finite_interval():
    pdf = BreitWigner1D(mean=5.279, width=0.030, low=5.15, high=5.40)
    x = jnp.linspace(5.15, 5.40, 40001)
    assert jnp.isclose(_trapz(pdf(x), x), 1.0, rtol=2e-4, atol=2e-4)


def test_relativistic_lineshape_intensity_uses_context_mass_range_and_is_normalized():
    context = ResonanceContext(
        parent_mass=5.27934,
        daughter_masses=(0.493677, 0.13957039),
        bachelor_mass=0.13957039,
        spin=1,
        pole_mass=0.8958,
        pole_width=0.0474,
        resonance_radius=4.0,
        parent_radius=4.0,
    )
    pdf = LineshapeIntensity1D.from_context(
        RelativisticBreitWigner(), context, quadrature_order=512
    )
    expected_low = 0.493677 + 0.13957039
    expected_high = 5.27934 - 0.13957039
    assert jnp.isclose(pdf.low, expected_low)
    assert jnp.isclose(pdf.high, expected_high)

    x = jnp.linspace(pdf.low, pdf.high, 60001)
    assert jnp.isclose(_trapz(pdf(x), x), 1.0, rtol=8e-4, atol=8e-4)
    assert jnp.all(pdf(x) >= 0.0)


def test_exponential_is_normalized_on_finite_interval():
    pdf = Exponential1D(slope=-1.3, low=0.0, high=1.0)
    x = jnp.linspace(0.0, 1.0, 20001)
    assert jnp.isclose(_trapz(pdf(x), x), 1.0, rtol=2e-4, atol=2e-4)


def test_histogram_is_normalized():
    pdf = Histogram1D(edges=jnp.array([0.0, 0.2, 0.5, 1.0]), values=jnp.array([1.0, 3.0, 2.0]))
    widths = jnp.diff(pdf.edges)
    assert jnp.isclose(jnp.sum(widths * pdf.values), 1.0)


def test_factorized_density_multiplies_independent_terms():
    base = lambda pars: jnp.array([0.2, 0.8])
    mass_pdf = Gaussian1D(mean=5.28, sigma=0.02, low=5.2, high=5.35)
    density = FactorizedDensity(
        base_density=base,
        observables={"mass": jnp.array([5.28, 5.30])},
        pdfs={"mass": mass_pdf},
    )
    result = density({})
    expected = base({}) * mass_pdf(jnp.array([5.28, 5.30]), {})
    assert jnp.allclose(result, expected)
