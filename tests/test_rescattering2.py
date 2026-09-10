import math

import jax.numpy as jnp

from dalitzplotfitter import Rescattering2, ResonanceContext, enable_x64


enable_x64()


def _context(spin=0):
    mk = 0.493677
    return ResonanceContext(
        parent_mass=5.27934,
        daughter_masses=(mk, mk),
        bachelor_mass=mk,
        spin=spin,
        pole_mass=1.5,
        pole_width=0.0,
    )


def test_rescattering2_matches_laura_reference_points():
    model = Rescattering2()
    context = _context()

    expected = {
        1.00: complex(-0.4072711909399807, -0.3666977380596356),
        1.20: complex(-0.19650139341624523, -0.03512873838827652),
        1.60: complex(-0.014107320439921921, -0.03513810923005649),
        1.80: complex(0.032860715369232144, -0.027130409606991458),
    }

    for mass, reference in expected.items():
        value = complex(model(jnp.asarray(mass), context))
        assert abs(value - reference) < 2e-12


def test_rescattering2_threshold_phase_is_226p5_degrees():
    model = Rescattering2()
    phase = float(model.phase(jnp.asarray(model.threshold_mass)))
    assert abs(phase - math.radians(226.5)) < 1e-12


def test_rescattering2_is_continuous_at_transition():
    model = Rescattering2()
    context = _context()
    eps = 1e-9

    left = complex(model(jnp.asarray(model.transition_mass - eps), context))
    at = complex(model(jnp.asarray(model.transition_mass), context))
    right = complex(model(jnp.asarray(model.transition_mass + eps), context))

    assert abs(left - at) < 2e-6
    assert abs(right - at) < 2e-6


def test_rescattering2_zero_at_upper_edge():
    model = Rescattering2()
    at_upper = complex(model(jnp.asarray(model.maximum_mass), _context()))
    assert abs(at_upper) == 0.0


def test_rescattering2_is_zero_below_kk_threshold():
    model = Rescattering2()
    context = _context()

    value = complex(model(jnp.asarray(0.90), context))
    assert abs(value) == 0.0


def test_rescattering2_raw_magnitude_extrapolates_below_kk_threshold():
    # g_00(m) itself (the raw Chebyshev magnitude) is still defined below
    # threshold; only resAmp's explicit mag=0 cut (applied in __call__)
    # removes it from the physical amplitude.
    model = Rescattering2()
    assert abs(float(model.magnitude(jnp.asarray(0.90)))) > 0.0


def test_rescattering2_suppression_factor_reduces_high_mass_amplitude():
    model = Rescattering2()
    context = _context()

    with_lambda = complex(model(jnp.asarray(1.6), context))
    unsuppressed = Rescattering2(lambda_scale=1.0e6)
    without_suppression = complex(unsuppressed(jnp.asarray(1.6), context))

    assert abs(with_lambda) < abs(without_suppression)


def test_rescattering2_zero_above_upper_domain():
    model = Rescattering2()
    value = complex(model(jnp.asarray(2.01), _context()))
    assert abs(value) == 0.0


def test_rescattering2_rejects_non_scalar_context():
    try:
        Rescattering2()(jnp.asarray(1.2), _context(spin=1))
    except ValueError as exc:
        assert "spin-0" in str(exc)
    else:
        raise AssertionError("Rescattering2 accepted a non-scalar context")
