import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.inverse_transform import DalitzInverseTransformSampler
from dalitzplotfitter.kinematics import dalitz_s13_limits


def _mass_squared(momentum):
    momentum = np.asarray(momentum)
    return momentum[:, 0] ** 2 - np.sum(momentum[:, 1:] ** 2, axis=1)


def test_inverse_transform_constant_density_matches_physical_dalitz_area():
    mother = 2.0
    masses = (0.2, 0.3, 0.4)
    sampler = DalitzInverseTransformSampler.prepare(
        mother,
        masses,
        lambda data: jnp.ones_like(data["s12"]),
        resolution=160,
    )
    sample = sampler.generate(20_000, seed=123)

    s12 = np.asarray(sample.s12)
    s13 = np.asarray(sample.s13)
    s23 = np.asarray(sample.s23)
    low, high = dalitz_s13_limits(
        sample.s12,
        mother_mass=mother,
        masses=masses,
    )
    assert np.all(s13 >= np.asarray(low) - 1e-11)
    assert np.all(s13 <= np.asarray(high) + 1e-11)

    invariant_sum = mother**2 + sum(mass**2 for mass in masses)
    assert np.allclose(s12 + s13 + s23, invariant_sum, rtol=0.0, atol=2e-10)

    grid = np.linspace((masses[0] + masses[1]) ** 2, (mother - masses[2]) ** 2, 20000)
    lo_grid, hi_grid = dalitz_s13_limits(
        jnp.asarray(grid),
        mother_mass=mother,
        masses=masses,
    )
    width = np.asarray(hi_grid - lo_grid)
    expected_mean = np.trapezoid(grid * width, grid) / np.trapezoid(width, grid)
    assert np.isclose(s12.mean(), expected_mean, atol=0.012)


def test_inverse_transform_reconstructed_momenta_are_on_shell_and_conserved():
    mother = 2.0
    masses = (0.2, 0.3, 0.4)
    sampler = DalitzInverseTransformSampler.prepare(
        mother,
        masses,
        lambda data: 1.0 + 0.3 * data["s12"] + 0.1 * data["s13"],
        resolution=128,
    )
    sample = sampler.generate(4000, seed=456)

    p1 = np.asarray(sample.p1)
    p2 = np.asarray(sample.p2)
    p3 = np.asarray(sample.p3)
    assert np.allclose(_mass_squared(p1), masses[0] ** 2, atol=2e-10)
    assert np.allclose(_mass_squared(p2), masses[1] ** 2, atol=2e-10)
    assert np.allclose(_mass_squared(p3), masses[2] ** 2, atol=5e-10)

    total = p1 + p2 + p3
    assert np.allclose(total[:, 0], mother, atol=2e-10)
    assert np.allclose(total[:, 1:], 0.0, atol=2e-10)
