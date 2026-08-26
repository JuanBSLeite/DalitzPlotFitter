import math

import sympy as sp

from dalitzplotfitter.dynamics import (
    blatt_weisskopf_factor,
    breakup_momentum,
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
