import jax.numpy as jnp
import pytest

from dalitzplotfitter import (
    DalitzGrid,
    DecayChannel,
    DecayModel,
    LauraGaussLegendreGrid,
    NonResonant,
    RealImag,
    enable_x64,
)
from dalitzplotfitter.integration import (
    GridIntegrator,
    matrix_normalization,
    normalization_matrix,
)

enable_x64()


MOTHER_MASS = 1.86966
MASSES = (0.13957, 0.13957, 0.13957)


def test_laura_grid_validates_configuration():
    with pytest.raises(ValueError, match="bin_width must be positive"):
        LauraGaussLegendreGrid(MOTHER_MASS, MASSES, bin_width=0.0)
    with pytest.raises(ValueError, match="order_m13 must be at least 2"):
        LauraGaussLegendreGrid(MOTHER_MASS, MASSES, order_m13=1)


def test_laura_weights_integrate_constant_to_dalitz_area():
    laura = LauraGaussLegendreGrid(
        MOTHER_MASS,
        MASSES,
        order_m13=500,
        order_m23=500,
    ).sample()
    reference_area = DalitzGrid(
        MOTHER_MASS,
        MASSES,
        resolution=100,
        boundary_resolution=100_001,
    ).area

    integral = GridIntegrator(laura).integrate(
        lambda data: jnp.ones_like(data["s12"])
    )
    assert jnp.isclose(integral, reference_area, rtol=2.0e-4, atol=0.0)
    assert bool(jnp.all(laura.weights > 0.0))
    invariant_sum = MOTHER_MASS**2 + sum(mass**2 for mass in MASSES)
    assert jnp.allclose(laura.s12 + laura.s13 + laura.s23, invariant_sum)


def test_laura_matrix_normalization_matches_direct_integral():
    sample = LauraGaussLegendreGrid(
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


def test_decay_model_selects_and_reuses_laura_normalization():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = DecayModel(
        channel,
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="laura",
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
