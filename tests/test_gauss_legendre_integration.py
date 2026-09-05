import jax.numpy as jnp
import pytest

from dalitzplotfitter.integration.adaptive_square_dalitz import AdaptiveSquareDalitzGrid

from dalitzplotfitter import (
    DalitzGaussLegendreGrid,
    DecayChannel,
    DecayModel,
    NonResonant,
    RealImag,
    SquareDalitzGrid,
    enable_x64,
)
from dalitzplotfitter.integration import (
    AdaptiveDalitzGaussLegendreGrid,
    GridIntegrator,
    matrix_normalization,
    normalization_matrix,
)

enable_x64()


MOTHER_MASS = 1.86966
MASSES = (0.13957, 0.13957, 0.13957)


def test_gauss_legendre_grid_validates_configuration():
    with pytest.raises(ValueError, match="bin_width must be positive"):
        DalitzGaussLegendreGrid(MOTHER_MASS, MASSES, bin_width=0.0)
    with pytest.raises(ValueError, match="order_m13 must be at least 2"):
        DalitzGaussLegendreGrid(MOTHER_MASS, MASSES, order_m13=1)


def test_gauss_legendre_weights_integrate_constant_to_square_dalitz_area():
    gauss_legendre = DalitzGaussLegendreGrid(
        MOTHER_MASS,
        MASSES,
        order_m13=500,
        order_m23=500,
    ).sample()
    square = SquareDalitzGrid(
        MOTHER_MASS,
        MASSES,
        resolution=300,
    ).sample()
    reference_area = jnp.mean(square.weights)

    integral = GridIntegrator(gauss_legendre).integrate(
        lambda data: jnp.ones_like(data["s12"])
    )
    assert jnp.isclose(integral, reference_area, rtol=2.0e-4, atol=0.0)
    assert bool(jnp.all(gauss_legendre.weights > 0.0))
    invariant_sum = MOTHER_MASS**2 + sum(mass**2 for mass in MASSES)
    assert jnp.allclose(
        gauss_legendre.s12 + gauss_legendre.s13 + gauss_legendre.s23,
        invariant_sum,
    )


def test_gauss_legendre_matrix_normalization_matches_direct_integral():
    sample = DalitzGaussLegendreGrid(
        MOTHER_MASS,
        MASSES,
        order_m13=160,
        order_m23=160,
    ).sample()
    components = jnp.stack(
        (
            jnp.ones_like(sample.s12, dtype=jnp.complex128),
            sample.s12 + 1j * sample.s13,
        ),
        axis=1,
    )
    coefficients = jnp.asarray([0.7 - 0.2j, -0.3 + 0.5j])
    matrix = normalization_matrix(components, sample.weights)
    direct = jnp.mean(
        sample.weights * jnp.abs(components @ coefficients) ** 2
    )
    assert jnp.allclose(
        matrix_normalization(coefficients, matrix),
        direct,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_decay_model_selects_and_reuses_gauss_legendre_normalization():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = DecayModel(
        channel,
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="gauss-legendre",
        normalization_order_m13=80,
        normalization_order_m23=70,
    )
    first = model.normalization_sample
    second = model.normalization_sample
    data = model.generate_phase_space(16, seed=91)
    cache = model.prepare_cache(data)

    assert first is second
    assert 0 < first.size < 80 * 70
    assert bool(jnp.any(first.weights != first.weights[0]))
    assert jnp.allclose(
        jnp.real(jnp.diag(cache.normalization_matrix_fixed)),
        jnp.ones((1,)),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_adaptive_grid_uses_laura_narrow_resonance_defaults():
    grid = AdaptiveDalitzGaussLegendreGrid(
        MOTHER_MASS,
        MASSES,
        m13_narrow_resonances=((0.78265, 0.00849),),
    )

    narrow_segments = [segment for segment in grid.m13_segments if segment.narrow]
    assert len(narrow_segments) == 1
    segment = narrow_segments[0]
    assert jnp.isclose(segment.low, 0.78265 - 5.0 * 0.00849)
    assert jnp.isclose(segment.high, 0.78265 + 5.0 * 0.00849)
    assert jnp.isclose(segment.target_width, 0.00849 / 100.0)
    assert 1000 <= segment.order <= 1001


def test_adaptive_grid_integrates_constant_with_global_weight_convention():
    reference = DalitzGaussLegendreGrid(
        MOTHER_MASS,
        MASSES,
        order_m13=260,
        order_m23=260,
    ).sample()
    adaptive = AdaptiveDalitzGaussLegendreGrid(
        MOTHER_MASS,
        MASSES,
        m13_narrow_resonances=((0.78, 0.012),),
        bin_width=0.08,
        narrow_width=0.020,
        window_n_widths=3.0,
        binning_factor=8.0,
    ).sample()

    reference_area = jnp.mean(reference.weights)
    adaptive_area = jnp.mean(adaptive.weights)
    assert jnp.allclose(adaptive_area, reference_area, rtol=8e-4, atol=1e-6)
    assert bool(jnp.all(adaptive.weights > 0.0))


def test_adaptive_grid_overlap_uses_finest_requested_binning():
    grid = AdaptiveDalitzGaussLegendreGrid(
        MOTHER_MASS,
        MASSES,
        m13_narrow_resonances=(
            (0.800, 0.012),
            (0.825, 0.006),
        ),
        bin_width=0.05,
        window_n_widths=5.0,
        binning_factor=20.0,
    )

    fine = min(segment.target_width for segment in grid.m13_segments if segment.narrow)
    assert jnp.isclose(fine, 0.006 / 20.0)


def test_adaptive_square_dalitz_integrates_constant_and_refines_narrow_band():
    reference = SquareDalitzGrid(
        MOTHER_MASS,
        MASSES,
        resolution=180,
        pair=(0, 2),
    ).sample()
    adaptive = AdaptiveSquareDalitzGrid(
        MOTHER_MASS,
        MASSES,
        narrow_resonances=(((0, 2), 0.78, 0.012),),
        resolution=24,
        pair=(0, 2),
        window_n_widths=3.0,
        binning_factor=4.0,
        cell_order=6,
        max_depth=6,
    ).sample()

    assert adaptive.size > 24**2
    assert bool(jnp.all(adaptive.weights > 0.0))
    assert jnp.allclose(
        jnp.mean(adaptive.weights),
        jnp.mean(reference.weights),
        rtol=2e-3,
        atol=2e-6,
    )
