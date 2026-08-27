import math

import jax.numpy as jnp

from dalitzplotfitter.dynamics import (
    RelativisticBreitWigner,
    ResonanceContext,
    blatt_weisskopf_from_momenta,
    breakup_momentum,
    covariant_spin_factor,
    energy_dependent_width,
)


def _context(spin=1):
    return ResonanceContext(
        parent_mass=1.86966,
        daughter_masses=(0.13957, 0.13957),
        bachelor_mass=0.13957,
        spin=spin,
        pole_mass=0.775,
        pole_width=0.149,
        resonance_radius=1.5,
        parent_radius=5.0,
    )


def test_breakup_momentum_matches_two_body_formula():
    m, m1, m2 = 1.2, 0.3, 0.2
    lam = m**4 + m1**4 + m2**4 - 2 * (
        m**2 * m1**2 + m**2 * m2**2 + m1**2 * m2**2
    )
    expected = math.sqrt(lam) / (2 * m)
    assert math.isclose(float(breakup_momentum(m, m1, m2)), expected, rel_tol=1e-6)


def test_covariant_spin_factors_match_reference_formulas():
    p_star, p, q, c, m_parent = 0.8, 1.1, 0.3, 0.25, 1.87
    r = p**2 / m_parent**2
    expected = {
        0: 1.0,
        1: -2 * p_star * q * math.sqrt(1 + r) * c,
        2: (4 / 3) * (p_star * q) ** 2 * (1.5 + r) * (3 * c**2 - 1),
        3: -(24 / 15) * (p_star * q) ** 3 * math.sqrt(1 + r)
        * (2.5 + r) * (5 * c**3 - 3 * c),
        4: (16 / 35) * (p_star * q) ** 4 * (8 * r**2 + 40 * r + 35)
        * (35 * c**4 - 30 * c**2 + 3),
    }
    for angular_momentum, target in expected.items():
        value = covariant_spin_factor(
            jnp.asarray(p_star), jnp.asarray(p), jnp.asarray(q), jnp.asarray(c),
            m_parent, angular_momentum,
        )
        assert math.isclose(float(value), target, rel_tol=1e-6, abs_tol=1e-7)


def test_blatt_weisskopf_is_one_at_pole():
    q0 = breakup_momentum(0.775, 0.13957, 0.13957)
    factor = blatt_weisskopf_from_momenta(q0, q0, 1, 1.5)
    assert math.isclose(float(factor), 1.0, rel_tol=1e-7, abs_tol=1e-7)


def test_running_width_equals_pole_width_at_pole():
    context = _context(spin=1)
    width = energy_dependent_width(context.pole_mass, context)
    assert math.isclose(
        float(width), context.pole_width, rel_tol=1e-6, abs_tol=1e-7
    )


def test_rbw_has_unit_numerator_at_pole():
    context = _context(spin=1)
    value = complex(RelativisticBreitWigner()(context.pole_mass, context))
    expected = 1j / (context.pole_mass * context.pole_width)
    assert math.isclose(value.real, expected.real, abs_tol=1e-6)
    assert math.isclose(value.imag, expected.imag, rel_tol=1e-6, abs_tol=1e-6)
