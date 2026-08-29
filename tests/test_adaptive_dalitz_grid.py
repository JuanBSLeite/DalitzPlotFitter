import jax.numpy as jnp

from dalitzplotfitter.kinematics import AdaptiveDalitzGrid, DalitzGrid


MASS = 1.86965
MASSES = (0.13957, 0.13957, 0.13957)


def test_adaptive_grid_preserves_constant_integral():
    adaptive = AdaptiveDalitzGrid(
        MASS,
        MASSES,
        base_resolution=8,
        max_depth=2,
        tolerance=0.05,
    )

    def probe(data):
        return jnp.exp(-((data["s12"] - 0.8) / 0.08) ** 2)

    result = adaptive.build((probe,))
    estimate = jnp.mean(result.sample.weights)
    reference_area = DalitzGrid(MASS, MASSES, resolution=200).area

    assert result.size >= 8**2
    assert jnp.any(result.depth > 0)
    assert jnp.isclose(estimate, reference_area, rtol=2e-6, atol=1e-10)


def test_adaptive_grid_refines_arbitrary_complex_probe():
    adaptive = AdaptiveDalitzGrid(
        MASS,
        MASSES,
        base_resolution=10,
        max_depth=2,
        tolerance=0.10,
    )

    def complex_probe(data):
        x = data["s12"]
        return 1.0 / (0.75 - x - 0.03j)

    result = adaptive.build((complex_probe,))

    assert result.size > 10**2
    assert int(jnp.max(result.depth)) == 2
    assert jnp.all(result.sample.weights > 0.0)
