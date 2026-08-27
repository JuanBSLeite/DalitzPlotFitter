import jax.numpy as jnp

from dalitzplotfitter.integration import (
    MonteCarloIntegrator,
    matrix_normalization,
    normalization_matrix,
)
from dalitzplotfitter.kinematics import PhaseSpaceSample


def test_monte_carlo_integrator_is_deterministic_for_fixed_sample():
    sample = PhaseSpaceSample(
        s12=jnp.linspace(0.1, 1.0, 4096),
        s13=jnp.linspace(0.2, 1.1, 4096),
        s23=jnp.linspace(0.3, 1.2, 4096),
        weights=jnp.linspace(0.5, 1.5, 4096),
    )
    integrator = MonteCarloIntegrator(sample)

    def function(data):
        return 1.0 + 0.2 * data["s12"]

    first = integrator.integrate(function)
    second = integrator.integrate(function)
    assert jnp.array_equal(first, second)
    assert first > 0.0


def test_normalization_matrix_matches_direct_intensity():
    components = jnp.asarray(
        [
            [1.0 + 0.0j, 0.5 + 0.2j],
            [0.8 - 0.1j, 1.2 + 0.3j],
            [1.1 + 0.4j, 0.7 - 0.2j],
        ]
    )
    weights = jnp.asarray([0.7, 1.1, 0.9])
    coefficients = jnp.asarray([1.2 + 0.3j, -0.4 + 0.8j])
    matrix = normalization_matrix(components, weights)
    matrix_value = matrix_normalization(coefficients, matrix)
    direct = jnp.mean(weights * jnp.abs(components @ coefficients) ** 2)
    assert jnp.allclose(matrix_value, direct)
