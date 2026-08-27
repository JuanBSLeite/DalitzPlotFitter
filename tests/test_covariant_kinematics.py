import math

import jax.numpy as jnp

from dalitzplotfitter.kinematics import (
    PhasespaceMC,
    boost_to_rest_frame,
    covariant_kinematics,
    invariant_mass_squared,
)


def _event():
    return PhasespaceMC(
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
    ).generate(1, seed=4137)


def test_boost_to_resonance_rest_frame_zeroes_resonance_momentum():
    sample = _event()
    resonance = sample.p1 + sample.p2
    boosted = boost_to_rest_frame(resonance, resonance)
    assert jnp.allclose(boosted[..., 1:], 0.0, atol=1e-8, rtol=0.0)
    assert jnp.allclose(
        boosted[..., 0] ** 2,
        invariant_mass_squared(resonance),
        atol=1e-8,
        rtol=1e-7,
    )


def test_covariant_momenta_match_two_body_invariant_formulae():
    sample = _event()
    values = covariant_kinematics(sample.p1, sample.p2, sample.p3)
    m_parent = 1.86966
    m_bachelor = 0.13957
    m_daughter = 0.13957
    m_res = float(values.resonance_mass[0])

    def kallen(x, y, z):
        return x * x + y * y + z * z - 2 * x * y - 2 * x * z - 2 * y * z

    root = math.sqrt(kallen(m_parent**2, m_res**2, m_bachelor**2))
    p_star_expected = root / (2 * m_parent)
    p_expected = root / (2 * m_res)
    q_expected = math.sqrt(
        kallen(m_res**2, m_daughter**2, m_daughter**2)
    ) / (2 * m_res)
    assert math.isclose(float(values.p_star[0]), p_star_expected, rel_tol=1e-6)
    assert math.isclose(float(values.p[0]), p_expected, rel_tol=1e-6)
    assert math.isclose(float(values.q[0]), q_expected, rel_tol=1e-6)
    assert -1.0 <= float(values.cos_theta[0]) <= 1.0


def test_equal_mass_daughter_exchange_flips_covariant_helicity_angle():
    sample = _event()
    first = covariant_kinematics(sample.p1, sample.p2, sample.p3)
    second = covariant_kinematics(sample.p2, sample.p1, sample.p3)
    assert jnp.allclose(first.resonance_mass, second.resonance_mass, atol=1e-7)
    assert jnp.allclose(first.p_star, second.p_star, atol=1e-7)
    assert jnp.allclose(first.p, second.p, atol=1e-7)
    assert jnp.allclose(first.q, second.q, atol=1e-7)
    assert jnp.allclose(first.cos_theta, -second.cos_theta, atol=1e-6)
