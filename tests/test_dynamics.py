import math

import jax.numpy as jnp

from dalitzplotfitter.dynamics import (
    RelativisticBreitWigner,
    ResonanceContext,
    ZemachP,
    ZemachPstar,
    Zemach_P,
    Zemach_Pstar,
    bachelor_momentum_resonance_frame,
    blatt_weisskopf_from_momenta,
    breakup_momentum,
    covariant_spin_factor,
    effective_pole_mass,
    energy_dependent_width,
    zemach_spin_factor,
)
from dalitzplotfitter.kinematics import CovariantKinematics


def _context(spin=1, *, pole_mass=0.775):
    return ResonanceContext(
        parent_mass=1.86966,
        daughter_masses=(0.13957, 0.13957),
        bachelor_mass=0.13957,
        spin=spin,
        pole_mass=pole_mass,
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


def test_bachelor_momentum_in_resonance_frame_matches_two_body_formula():
    parent, resonance, bachelor = 1.87, 0.775, 0.140
    lam = parent**4 + resonance**4 + bachelor**4 - 2 * (
        parent**2 * resonance**2
        + parent**2 * bachelor**2
        + resonance**2 * bachelor**2
    )
    expected = math.sqrt(lam) / (2 * resonance)
    value = bachelor_momentum_resonance_frame(parent, resonance, bachelor)
    assert math.isclose(float(value), expected, rel_tol=1e-7)


def test_effective_pole_mass_matches_reference_virtual_state_mapping():
    context = _context(pole_mass=2.2)
    minimum = sum(context.daughter_masses)
    maximum = context.parent_mass - context.bachelor_mass
    span = maximum - minimum
    midpoint = 0.5 * (minimum + maximum)
    expected = minimum + 0.5 * span * (
        1.0 + math.tanh((context.pole_mass - midpoint) / span)
    )
    value = float(effective_pole_mass(context))
    assert minimum < value < maximum
    assert math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_effective_pole_mass_is_unchanged_inside_physical_range():
    context = _context(pole_mass=0.775)
    assert math.isclose(
        float(effective_pole_mass(context)),
        context.pole_mass,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


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


def test_zemach_spin_factors_match_laura_reference_formulas():
    momentum, q, c = 1.1, 0.3, 0.25
    expected = {
        0: 1.0,
        1: -2.0 * momentum * q * c,
        2: (4.0 / 3.0) * (momentum * q) ** 2 * (3.0 * c**2 - 1.0),
        3: -(24.0 / 15.0) * (momentum * q) ** 3 * (5.0 * c**3 - 3.0 * c),
        4: (16.0 / 35.0)
        * (momentum * q) ** 4
        * (35.0 * c**4 - 30.0 * c**2 + 3.0),
    }
    for angular_momentum, target in expected.items():
        value = zemach_spin_factor(
            jnp.asarray(momentum),
            jnp.asarray(q),
            jnp.asarray(c),
            angular_momentum,
        )
        assert math.isclose(float(value), target, rel_tol=1e-6, abs_tol=1e-7)


def test_zemach_p_and_pstar_select_the_expected_bachelor_momentum():
    kin = CovariantKinematics(
        resonance_mass=jnp.asarray(0.9),
        p_star=jnp.asarray(0.7),
        p=jnp.asarray(1.1),
        q=jnp.asarray(0.3),
        cos_theta=jnp.asarray(0.25),
    )
    context = _context(spin=2)

    expected_p = zemach_spin_factor(kin.p, kin.q, kin.cos_theta, 2)
    expected_pstar = zemach_spin_factor(kin.p_star, kin.q, kin.cos_theta, 2)

    assert math.isclose(
        float(ZemachP()(kin, context)),
        float(expected_p),
        rel_tol=1e-7,
        abs_tol=1e-7,
    )
    assert math.isclose(
        float(ZemachPstar()(kin, context)),
        float(expected_pstar),
        rel_tol=1e-7,
        abs_tol=1e-7,
    )
    assert Zemach_P is ZemachP
    assert Zemach_Pstar is ZemachPstar


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


def test_virtual_pole_width_is_finite_inside_dalitz_region():
    context = _context(spin=1, pole_mass=2.2)
    width = energy_dependent_width(jnp.asarray(1.0), context)
    assert bool(jnp.isfinite(width))
    assert float(width) > 0.0


def test_rbw_has_unit_numerator_at_pole():
    context = _context(spin=1)
    value = complex(RelativisticBreitWigner()(context.pole_mass, context))
    expected = 1j / (context.pole_mass * context.pole_width)
    assert math.isclose(value.real, expected.real, abs_tol=1e-6)
    assert math.isclose(value.imag, expected.imag, rel_tol=1e-6, abs_tol=1e-6)
