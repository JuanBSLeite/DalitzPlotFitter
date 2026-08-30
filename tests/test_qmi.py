import jax.numpy as jnp

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    Parameter,
    ParameterKind,
    QMI,
    RealImag,
    Resonance,
    ResonanceContext,
    enable_x64,
)


enable_x64()


def _context():
    mpi = 0.13957039
    return ResonanceContext(
        parent_mass=1.86965,
        daughter_masses=(mpi, mpi),
        bachelor_mass=mpi,
        spin=0,
        pole_mass=1.0,
        pole_width=0.0,
    )


def test_qmi_returns_exact_complex_values_at_knots():
    knots = (0.30, 0.60, 0.90)
    magnitudes = (1.0, 2.0, 3.0)
    phases = (0.0, 0.5, 1.0)
    model = QMI(knots=knots, magnitudes=magnitudes, phases=phases)
    values = model(jnp.asarray(knots), _context())
    expected = jnp.asarray(magnitudes) * jnp.exp(1j * jnp.asarray(phases))
    assert bool(jnp.allclose(values, expected, rtol=0.0, atol=1e-12))


def test_qmi_interpolates_magnitude_and_phase_linearly():
    model = QMI(
        knots=(0.4, 0.8),
        magnitudes=(1.0, 3.0),
        phases=(0.2, 1.0),
    )
    magnitude, phase = model.interpolated_magnitude_phase(jnp.asarray(0.6))
    assert abs(float(magnitude) - 2.0) < 1e-12
    assert abs(float(phase) - 0.6) < 1e-12


def test_qmi_clamps_to_endpoint_values_outside_knot_range():
    model = QMI(
        knots=(0.4, 0.8),
        magnitudes=(1.0, 3.0),
        phases=(0.2, 1.0),
    )
    low_mag, low_phase = model.interpolated_magnitude_phase(jnp.asarray(0.2))
    high_mag, high_phase = model.interpolated_magnitude_phase(jnp.asarray(1.0))
    assert abs(float(low_mag) - 1.0) < 1e-12
    assert abs(float(low_phase) - 0.2) < 1e-12
    assert abs(float(high_mag) - 3.0) < 1e-12
    assert abs(float(high_phase) - 1.0) < 1e-12


def test_qmi_rejects_non_scalar_context():
    context = _context()
    bad = ResonanceContext(
        parent_mass=context.parent_mass,
        daughter_masses=context.daughter_masses,
        bachelor_mass=context.bachelor_mass,
        spin=1,
        pole_mass=context.pole_mass,
        pole_width=context.pole_width,
    )
    model = QMI(knots=(0.4, 0.8), magnitudes=(1.0, 1.0), phases=(0.0, 0.0))
    try:
        model(jnp.asarray(0.6), bad)
    except ValueError as exc:
        assert "scalar" in str(exc)
    else:
        raise AssertionError("QMI accepted a non-scalar context")


def test_qmi_knot_parameters_are_collected_and_resolved_by_decay_model():
    owner = "pipi_S_qmi"
    a0 = Parameter(
        name="qmi_a0",
        value=1.0,
        kind=ParameterKind.DYNAMICS,
        owner=owner,
    )
    d0 = Parameter(
        name="qmi_d0",
        value=0.0,
        kind=ParameterKind.DYNAMICS,
        owner=owner,
    )
    qmi = QMI(
        knots=(0.30, 0.60),
        magnitudes=(a0, 1.5),
        phases=(d0, 0.4),
    )
    decay = DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [
            Resonance(
                owner,
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=1.0,
                width=0.0,
                spin=0,
                lineshape=qmi,
            )
        ],
        normalization_resolution=30,
    )
    names = {parameter.name for parameter in decay.parameters}
    assert {"qmi_a0", "qmi_d0"}.issubset(names)

    data = decay.normalization_sample.as_dict()
    nominal = decay.intensity(data, {"qmi_a0": 1.0, "qmi_d0": 0.0})
    shifted = decay.intensity(data, {"qmi_a0": 2.0, "qmi_d0": 0.3})
    assert bool(jnp.any(jnp.abs(nominal - shifted) > 1e-10))
