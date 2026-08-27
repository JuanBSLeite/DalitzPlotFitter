import math

import sympy as sp

from dalitzplotfitter.dynamics import (
    bachelor_momentum_parent_frame,
    bachelor_momentum_resonance_frame,
    blatt_weisskopf_factor,
    breakup_momentum,
    covariant_angular_factor,
    energy_dependent_width,
    relativistic_breit_wigner,
)


def test_breakup_momentum_matches_two_body_formula():
    m = 1.2
    m1 = 0.3
    m2 = 0.2
    q = float(sp.N(breakup_momentum(m, m1, m2)))
    lam = m**4 + m1**4 + m2**4 - 2 * (m**2 * m1**2 + m**2 * m2**2 + m1**2 * m2**2)
    expected = math.sqrt(lam) / (2 * m)
    assert math.isclose(q, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_bachelor_momenta_use_parent_and_resonance_denominators():
    m_parent = 1.86966
    m_res = 0.775
    m_bachelor = 0.13957
    lam = (
        m_parent**4
        + m_res**4
        + m_bachelor**4
        - 2 * (m_parent**2 * m_res**2 + m_parent**2 * m_bachelor**2 + m_res**2 * m_bachelor**2)
    )
    root = math.sqrt(lam)
    p_star = float(sp.N(bachelor_momentum_parent_frame(m_parent, m_res, m_bachelor)))
    p = float(sp.N(bachelor_momentum_resonance_frame(m_parent, m_res, m_bachelor)))
    assert math.isclose(p_star, root / (2 * m_parent), rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(p, root / (2 * m_res), rel_tol=1e-12, abs_tol=1e-12)


def test_covariant_spin_zero_is_unity():
    value = covariant_angular_factor(0.8, 1.1, 0.3, -0.7, 1.87, 0)
    assert value == 1


def test_covariant_spin_one_matches_laura_formula():
    p_star = 0.8
    p = 1.1
    q = 0.3
    cos_theta = 0.4
    m_parent = 1.87
    value = float(
        sp.N(
            covariant_angular_factor(
                p_star,
                p,
                q,
                cos_theta,
                m_parent,
                1,
            )
        )
    )
    expected = -2 * p_star * q * math.sqrt(1 + p**2 / m_parent**2) * cos_theta
    assert math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_covariant_spin_two_matches_laura_formula():
    p_star = 0.8
    p = 1.1
    q = 0.3
    cos_theta = -0.35
    m_parent = 1.87
    value = float(
        sp.N(
            covariant_angular_factor(
                p_star,
                p,
                q,
                cos_theta,
                m_parent,
                2,
            )
        )
    )
    expected = (
        (4 / 3)
        * (p_star * q) ** 2
        * (1.5 + p**2 / m_parent**2)
        * (3 * cos_theta**2 - 1)
    )
    assert math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_covariant_spin_three_matches_laura_formula():
    p_star = 0.8
    p = 1.1
    q = 0.3
    cos_theta = 0.25
    m_parent = 1.87
    value = float(
        sp.N(
            covariant_angular_factor(
                p_star,
                p,
                q,
                cos_theta,
                m_parent,
                3,
            )
        )
    )
    r = p**2 / m_parent**2
    expected = (
        -(24 / 15)
        * (p_star * q) ** 3
        * math.sqrt(1 + r)
        * (2.5 + r)
        * (5 * cos_theta**3 - 3 * cos_theta)
    )
    assert math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_covariant_spin_four_matches_laura_formula():
    p_star = 0.8
    p = 1.1
    q = 0.3
    cos_theta = -0.2
    m_parent = 1.87
    value = float(
        sp.N(
            covariant_angular_factor(
                p_star,
                p,
                q,
                cos_theta,
                m_parent,
                4,
            )
        )
    )
    r = p**2 / m_parent**2
    expected = (
        (16 / 35)
        * (p_star * q) ** 4
        * (8 * r**2 + 40 * r + 35)
        * (35 * cos_theta**4 - 30 * cos_theta**2 + 3)
    )
    assert math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_running_width_equals_pole_width_at_pole():
    m0 = 0.775
    gamma0 = 0.149
    m_pi = 0.13957
    width = energy_dependent_width(
        mass=m0,
        mass0=m0,
        gamma0=gamma0,
        daughter_mass1=m_pi,
        daughter_mass2=m_pi,
        angular_momentum=1,
        meson_radius=1.5,
    )
    assert math.isclose(float(sp.N(width)), gamma0, rel_tol=1e-12, abs_tol=1e-12)


def test_blatt_weisskopf_is_one_at_pole():
    factor = blatt_weisskopf_factor(
        mass=0.775,
        mass0=0.775,
        daughter_mass1=0.13957,
        daughter_mass2=0.13957,
        angular_momentum=1,
        meson_radius=1.5,
    )
    assert math.isclose(float(sp.N(factor)), 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_laura_rbw_has_unit_numerator_convention_at_pole():
    m0 = 0.775
    gamma0 = 0.149
    value = complex(
        sp.N(
            relativistic_breit_wigner(
                mass=m0,
                mass0=m0,
                gamma0=gamma0,
                daughter_mass1=0.13957,
                daughter_mass2=0.13957,
                angular_momentum=1,
                meson_radius=1.5,
            )
        )
    )
    expected = 1j / (m0 * gamma0)
    assert math.isclose(value.real, expected.real, abs_tol=1e-12)
    assert math.isclose(value.imag, expected.imag, rel_tol=1e-12, abs_tol=1e-12)
