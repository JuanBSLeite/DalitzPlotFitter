import math

import jax.numpy as jnp

from dalitzplotfitter import (
    GooFitLegacyAngular,
    QMI,
    ResonanceContext,
    enable_x64,
)
from dalitzplotfitter.dynamics import (
    bachelor_momentum_parent_frame,
    bachelor_momentum_resonance_frame,
)
from dalitzplotfitter.kinematics import CovariantKinematics


enable_x64()


def _context(spin=0):
    return ResonanceContext(
        parent_mass=1.96834,
        daughter_masses=(0.13957, 0.13957),
        bachelor_mass=0.13957,
        spin=spin,
        pole_mass=1.0,
        pole_width=0.1,
    )


def test_linear_cartesian_qmi_matches_legacy_complex_interpolation():
    qmi = QMI(
        knots=(1.0, 2.0),
        magnitudes=(1.0, 1.0),
        phases=(0.0, math.pi),
        interpolation="linear-cartesian",
    )
    midpoint_mass = math.sqrt(0.5 * (1.0**2 + 2.0**2))
    value = qmi(jnp.asarray(midpoint_mass), _context())
    assert jnp.allclose(value, 0.0 + 0.0j, atol=1e-12, rtol=0.0)


def test_linear_cartesian_qmi_is_zero_outside_legacy_knot_range():
    qmi = QMI(
        knots=(1.0, 2.0),
        magnitudes=(2.0, 3.0),
        phases=(0.1, 0.2),
        interpolation="linear-cartesian",
    )
    assert jnp.allclose(qmi(jnp.asarray(0.9), _context()), 0.0 + 0.0j)
    assert jnp.allclose(qmi(jnp.asarray(2.1), _context()), 0.0 + 0.0j)


def test_goofit_legacy_angular_matches_dsppp_spin_factors():
    kin = CovariantKinematics(
        resonance_mass=jnp.asarray(1.1),
        p_star=jnp.asarray(0.4),
        p=jnp.asarray(0.7),
        q=jnp.asarray(0.3),
        cos_theta=jnp.asarray(0.25),
    )
    angular = GooFitLegacyAngular()

    spin1 = angular(kin, _context(spin=1))
    assert jnp.allclose(spin1, 4.0 * 0.7 * 0.3 * 0.25, atol=1e-12)

    spin2 = angular(kin, _context(spin=2))
    expected2 = (16.0 / 3.0) * (0.7 * 0.3) ** 2 * (3.0 * 0.25**2 - 1.0)
    assert jnp.allclose(spin2, expected2, atol=1e-12)


def test_parent_and_resonance_frame_bachelor_momenta_are_distinct():
    parent = 1.96834
    resonance = 1.27
    bachelor = 0.13957
    p_parent = bachelor_momentum_parent_frame(parent, resonance, bachelor)
    p_res = bachelor_momentum_resonance_frame(parent, resonance, bachelor)
    assert jnp.allclose(p_res / p_parent, parent / resonance, rtol=1e-12, atol=1e-12)
