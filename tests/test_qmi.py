import jax
import jax.numpy as jnp

from dalitzplotfitter.dynamics.lineshape.qmi import (
    _cubic_qmi_prepared,
    _hermite_qmi_prepared,
    _linear_cartesian_qmi_prepared,
)

from dalitzplotfitter import (
    QMI,
    DecayChannel,
    DecayModel,
    NonResonant,
    Parameter,
    ParameterKind,
    RealImag,
    Resonance,
    ResonanceContext,
    enable_x64,
)

enable_x64()


def _context():
    mpi = 0.13957039
    return ResonanceContext(
        parent_mass=1.96834,
        daughter_masses=(mpi, mpi),
        bachelor_mass=mpi,
        spin=0,
        pole_mass=1.0,
        pole_width=0.0,
    )


def test_qmi_returns_exact_complex_values_at_knots_for_both_interpolations():
    knots = (0.30, 0.60, 0.90, 1.20)
    magnitudes = (1.0, 2.0, 1.4, 3.0)
    phases = (0.0, 0.5, 0.8, 1.0)
    expected = jnp.asarray(magnitudes) * jnp.exp(1j * jnp.asarray(phases))
    for interpolation in ("linear", "cubic", "hermite"):
        model = QMI(
            knots=knots,
            magnitudes=magnitudes,
            phases=phases,
            interpolation=interpolation,
        )
        values = model(jnp.asarray(knots), _context())
        assert bool(jnp.allclose(values, expected, rtol=0.0, atol=1e-11))


def test_qmi_interpolates_magnitude_and_phase_linearly_in_s():
    model = QMI(
        knots=(0.4, 0.8),
        magnitudes=(1.0, 3.0),
        phases=(0.2, 1.0),
        interpolation="linear",
    )
    mass = jnp.sqrt(0.5 * (0.4**2 + 0.8**2))
    magnitude, phase = model.interpolated_magnitude_phase(mass)
    assert abs(float(magnitude) - 2.0) < 1e-12
    assert abs(float(phase) - 0.6) < 1e-12


def test_qmi_interpolates_real_and_imaginary_parts_linearly_in_s():
    model = QMI(
        knots=(0.4, 0.8),
        real_parts=(1.0, 3.0),
        imaginary_parts=(-2.0, 2.0),
        interpolation="linear",
    )
    mass = jnp.sqrt(0.5 * (0.4**2 + 0.8**2))
    value = model(mass, _context())
    real, imaginary = model.interpolated_cartesian(mass)
    magnitude, phase = model.interpolated_magnitude_phase(mass)

    assert abs(complex(value) - (2.0 + 0.0j)) < 1e-12
    assert abs(float(real) - 2.0) < 1e-12
    assert abs(float(imaginary)) < 1e-12
    assert abs(float(magnitude) - 2.0) < 1e-12
    assert abs(float(phase)) < 1e-12


def test_qmi_cartesian_returns_exact_complex_values_at_knots():
    knots = (0.30, 0.60, 0.90, 1.20)
    real_parts = (1.0, -0.5, 0.3, 2.0)
    imaginary_parts = (0.2, 1.5, -0.7, 0.0)
    expected = jnp.asarray(real_parts) + 1j * jnp.asarray(imaginary_parts)

    for interpolation in ("linear", "cubic", "hermite"):
        model = QMI(
            knots=knots,
            real_parts=real_parts,
            imaginary_parts=imaginary_parts,
            interpolation=interpolation,
        )
        values = model(jnp.asarray(knots), _context())
        assert bool(jnp.allclose(values, expected, rtol=0.0, atol=1e-11))


def test_qmi_requires_exactly_one_complete_parameterization():
    invalid = (
        {},
        {"magnitudes": (1.0, 1.0)},
        {"real_parts": (1.0, 1.0)},
        {
            "magnitudes": (1.0, 1.0),
            "phases": (0.0, 0.0),
            "real_parts": (1.0, 1.0),
            "imaginary_parts": (0.0, 0.0),
        },
    )
    for values in invalid:
        try:
            QMI(knots=(0.4, 0.8), **values)
        except ValueError as exc:
            assert "polar or Cartesian" in str(exc) or "same length" in str(exc)
        else:
            raise AssertionError(
                "QMI accepted an incomplete or ambiguous parameter set"
            )


def test_qmi_cubic_is_smooth_and_differs_from_linear_between_knots():
    knots = (0.3, 0.6, 0.9, 1.2)
    magnitudes = (1.0, 2.2, 0.9, 1.8)
    phases = (0.0, 0.7, 1.3, 1.8)
    mass = jnp.asarray(0.75)
    linear = QMI(knots, magnitudes, phases, interpolation="linear")
    cubic = QMI(knots, magnitudes, phases, interpolation="cubic")
    linear_mag, linear_phase = linear.interpolated_magnitude_phase(mass)
    cubic_mag, cubic_phase = cubic.interpolated_magnitude_phase(mass)
    assert abs(float(cubic_mag - linear_mag)) > 1e-6
    assert abs(float(cubic_phase - linear_phase)) > 1e-6


def test_qmi_clamps_to_endpoint_values_outside_knot_range_for_both_modes():
    for interpolation in ("linear", "cubic", "hermite"):
        model = QMI(
            knots=(0.4, 0.6, 0.8),
            magnitudes=(1.0, 2.0, 3.0),
            phases=(0.2, 0.5, 1.0),
            interpolation=interpolation,
        )
        low_mag, low_phase = model.interpolated_magnitude_phase(jnp.asarray(0.2))
        high_mag, high_phase = model.interpolated_magnitude_phase(jnp.asarray(1.0))
        assert abs(float(low_mag) - 1.0) < 1e-11
        assert abs(float(low_phase) - 0.2) < 1e-11
        assert abs(float(high_mag) - 3.0) < 1e-11
        assert abs(float(high_phase) - 1.0) < 1e-11


def test_qmi_rejects_invalid_interpolation():
    try:
        QMI(
            knots=(0.4, 0.6, 0.8),
            magnitudes=(1.0, 1.0, 1.0),
            phases=(0.0, 0.0, 0.0),
            interpolation="quadratic",
        )
    except ValueError as exc:
        assert all(name in str(exc) for name in ("linear", "cubic", "hermite"))
    else:
        raise AssertionError("QMI accepted an unsupported interpolation mode")


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
        knots=(0.30, 0.60, 0.90),
        magnitudes=(a0, 1.5, 1.2),
        phases=(d0, 0.4, 0.8),
        interpolation="cubic",
    )
    decay = DecayModel(
        DecayChannel("D_s+", ("pi-", "pi+", "pi+")),
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


def test_cartesian_qmi_parameters_are_collected_and_resolved_by_decay_model():
    owner = "pipi_S_cartesian_qmi"
    x0 = Parameter.dynamics("qmi_x0", 1.0, owner=owner)
    y0 = Parameter.dynamics("qmi_y0", 0.0, owner=owner)
    qmi = QMI(
        knots=(0.30, 0.60, 0.90),
        real_parts=(x0, 1.5, 1.2),
        imaginary_parts=(y0, 0.4, 0.8),
        interpolation="linear",
    )
    decay = DecayModel(
        DecayChannel("D_s+", ("pi-", "pi+", "pi+")),
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
    assert {"qmi_x0", "qmi_y0"}.issubset(names)

    data = decay.normalization_sample.as_dict()
    nominal = decay.intensity(data, {"qmi_x0": 1.0, "qmi_y0": 0.0})
    shifted = decay.intensity(data, {"qmi_x0": 2.0, "qmi_y0": 0.3})
    assert bool(jnp.any(jnp.abs(nominal - shifted) > 1e-10))


def test_qmi_linear_matches_jnp_interp_and_has_finite_gradients():
    knots = (0.30, 0.60, 0.90, 1.20)
    knot_s = jnp.asarray(knots) ** 2
    values = jnp.asarray((1.0, 2.0, 1.4, 3.0))
    masses = jnp.linspace(0.20, 1.30, 101)
    s = masses**2

    model = QMI(
        knots=knots,
        magnitudes=tuple(values),
        phases=(0.0, 0.0, 0.0, 0.0),
        interpolation="linear",
    )
    magnitude, _ = model.interpolated_magnitude_phase(masses)
    expected = jnp.interp(s, knot_s, values)
    assert jnp.allclose(magnitude, expected, rtol=1e-13, atol=1e-13)

    def objective(fp):
        local = QMI(
            knots=knots,
            magnitudes=tuple(fp),
            phases=(0.0, 0.0, 0.0, 0.0),
            interpolation="linear",
        )
        mag, _ = local.interpolated_magnitude_phase(masses)
        return jnp.sum(mag**2)

    gradient = jax.grad(objective)(values)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_prepared_cubic_qmi_matches_reference_value_and_gradient():
    knots = (0.30, 0.48, 0.67, 0.91, 1.20)
    knot_s = jnp.asarray(knots) ** 2
    values = jnp.asarray((1.0, -0.4, 1.7, 0.2, 1.1))
    masses = jnp.linspace(0.31, 1.19, 173)
    s = masses**2

    reference_model = QMI(
        knots=knots,
        magnitudes=tuple(values),
        phases=(0.0,) * len(knots),
        interpolation="cubic",
    )

    index = jnp.clip(
        jnp.searchsorted(knot_s, s, side="right") - 1,
        0,
        len(knots) - 2,
    )
    x0 = knot_s[index]
    x1 = knot_s[index + 1]
    fraction = (s - x0) / (x1 - x0)
    order = jnp.argsort(index).astype(jnp.int32)
    counts = jnp.bincount(index.astype(jnp.int32), length=len(knots) - 1)
    ends = jnp.cumsum(counts).astype(jnp.int32)
    starts = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), ends[:-1]))

    def prepared_objective(fp):
        interpolated = _cubic_qmi_prepared(
            fp,
            index,
            fraction,
            order,
            starts,
            ends,
        )
        return jnp.sum((1.0 + s) * interpolated**2)

    def reference_objective(fp):
        local = QMI(
            knots=knots,
            magnitudes=tuple(fp),
            phases=(0.0,) * len(knots),
            interpolation="cubic",
        )
        magnitude, _ = local.interpolated_magnitude_phase(masses)
        return jnp.sum((1.0 + s) * magnitude**2)

    prepared_value = prepared_objective(values)
    reference_value = reference_objective(values)
    prepared_gradient = jax.grad(prepared_objective)(values)
    reference_gradient = jax.grad(reference_objective)(values)

    assert jnp.allclose(prepared_value, reference_value, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(
        prepared_gradient,
        reference_gradient,
        rtol=2e-11,
        atol=2e-11,
    )


def test_prepared_cubic_cartesian_qmi_gradient_is_finite():
    knots = (0.30, 0.48, 0.67, 0.91, 1.20)
    knot_s = jnp.asarray(knots) ** 2
    masses = jnp.linspace(0.31, 1.19, 173)
    s = masses**2
    real = jnp.asarray((1.0, -0.4, 1.7, 0.2, 1.1))
    imaginary = jnp.asarray((0.2, 0.8, -0.6, 1.2, -0.1))

    model = QMI(
        knots=knots,
        real_parts=tuple(real),
        imaginary_parts=tuple(imaginary),
        interpolation="cubic",
    )
    index = jnp.clip(
        jnp.searchsorted(knot_s, s, side="right") - 1,
        0,
        len(knots) - 2,
    )
    x0 = knot_s[index]
    x1 = knot_s[index + 1]
    fraction = (s - x0) / (x1 - x0)
    order = jnp.argsort(index).astype(jnp.int32)
    counts = jnp.bincount(index.astype(jnp.int32), length=len(knots) - 1)
    ends = jnp.cumsum(counts).astype(jnp.int32)
    starts = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), ends[:-1]))

    def objective(real_values, imaginary_values):
        real_interp = _cubic_qmi_prepared(
            real_values,
            index,
            fraction,
            order,
            starts,
            ends,
        )
        imaginary_interp = _cubic_qmi_prepared(
            imaginary_values,
            index,
            fraction,
            order,
            starts,
            ends,
        )
        amplitude = real_interp + 1j * imaginary_interp
        return jnp.sum((1.0 + s) * jnp.abs(amplitude) ** 2)

    gradients = jax.grad(objective, argnums=(0, 1))(real, imaginary)
    assert bool(jnp.all(jnp.isfinite(gradients[0])))
    assert bool(jnp.all(jnp.isfinite(gradients[1])))


def test_qmi_local_cubic_uses_only_the_two_adjacent_knots():
    knots = (0.30, 0.60, 0.90, 1.20)
    mass = jnp.asarray(0.50)

    first = QMI(
        knots=knots,
        real_parts=(1.0, 2.0, 100.0, -50.0),
        imaginary_parts=(0.2, -0.4, 80.0, 90.0),
        interpolation="cubic",
    )
    second = QMI(
        knots=knots,
        real_parts=(1.0, 2.0, -999.0, 777.0),
        imaginary_parts=(0.2, -0.4, -333.0, 444.0),
        interpolation="cubic",
    )

    assert jnp.allclose(
        first(mass, _context()),
        second(mass, _context()),
        rtol=0.0,
        atol=1e-13,
    )


def test_qmi_local_cubic_supports_two_knots():
    model = QMI(
        knots=(0.4, 0.8),
        real_parts=(1.0, 3.0),
        imaginary_parts=(-2.0, 2.0),
        interpolation="cubic",
    )
    mass = jnp.sqrt(0.25 * 0.4**2 + 0.75 * 0.8**2)
    fraction = 0.75
    weight = 3.0 * fraction**2 - 2.0 * fraction**3
    expected = (1.0 + weight * 2.0) + 1j * (-2.0 + weight * 4.0)
    assert abs(complex(model(mass, _context())) - expected) < 1e-12


def test_qmi_hermite_is_local_to_neighboring_knots():
    knots = (0.30, 0.50, 0.70, 0.90, 1.10, 1.30)
    mass = jnp.asarray(0.80)

    first = QMI(
        knots=knots,
        real_parts=(1000.0, 1.0, 2.0, 3.0, 4.0, -1000.0),
        imaginary_parts=(-500.0, 0.2, -0.3, 0.4, -0.1, 700.0),
        interpolation="hermite",
    )
    second = QMI(
        knots=knots,
        real_parts=(-999.0, 1.0, 2.0, 3.0, 4.0, 888.0),
        imaginary_parts=(333.0, 0.2, -0.3, 0.4, -0.1, -444.0),
        interpolation="hermite",
    )

    # The interval [0.70, 0.90] uses knots 0.50, 0.70, 0.90, and 1.10.
    # Changing more distant knots must have exactly no effect.
    assert jnp.allclose(
        first(mass, _context()),
        second(mass, _context()),
        rtol=0.0,
        atol=1e-13,
    )


def test_prepared_hermite_qmi_matches_reference_value_and_gradient():
    knots = (0.30, 0.48, 0.67, 0.91, 1.20)
    knot_s = jnp.asarray(knots) ** 2
    values = jnp.asarray((1.0, -0.4, 1.7, 0.2, 1.1))
    masses = jnp.linspace(0.31, 1.19, 173)
    s = masses**2

    index = jnp.clip(
        jnp.searchsorted(knot_s, s, side="right") - 1,
        0,
        len(knots) - 2,
    )
    x0 = knot_s[index]
    x1 = knot_s[index + 1]
    fraction = (s - x0) / (x1 - x0)
    order = jnp.argsort(index).astype(jnp.int32)
    counts = jnp.bincount(index.astype(jnp.int32), length=len(knots) - 1)
    ends = jnp.cumsum(counts).astype(jnp.int32)
    starts = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), ends[:-1]))

    def prepared_objective(fp):
        interpolated = _hermite_qmi_prepared(
            fp,
            index,
            fraction,
            order,
            starts,
            ends,
            knot_s,
        )
        return jnp.sum((1.0 + s) * interpolated**2)

    def reference_objective(fp):
        local = QMI(
            knots=knots,
            magnitudes=tuple(fp),
            phases=(0.0,) * len(knots),
            interpolation="hermite",
        )
        magnitude, _ = local.interpolated_magnitude_phase(masses)
        return jnp.sum((1.0 + s) * magnitude**2)

    prepared_value = prepared_objective(values)
    reference_value = reference_objective(values)
    prepared_gradient = jax.grad(prepared_objective)(values)
    reference_gradient = jax.grad(reference_objective)(values)

    assert jnp.allclose(prepared_value, reference_value, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(
        prepared_gradient,
        reference_gradient,
        rtol=2e-11,
        atol=2e-11,
    )


def test_qmi_hermite_has_continuous_first_derivative_at_internal_knots():
    knots = (0.30, 0.50, 0.75, 1.05)
    values = (1.0, -0.4, 1.8, 0.2)
    model = QMI(
        knots=knots,
        real_parts=values,
        imaginary_parts=(0.0,) * len(knots),
        interpolation="hermite",
    )

    knot = knots[1]
    eps = 1e-7
    left = (
        jnp.real(model(jnp.asarray(knot), _context()))
        - jnp.real(model(jnp.asarray(knot - eps), _context()))
    ) / eps
    right = (
        jnp.real(model(jnp.asarray(knot + eps), _context()))
        - jnp.real(model(jnp.asarray(knot), _context()))
    ) / eps
    assert abs(float(left - right)) < 2e-4


def test_prepared_linear_cartesian_qmi_matches_reference_value_and_gradient():
    knots = (0.30, 0.48, 0.67, 0.91, 1.20)
    knot_s = jnp.asarray(knots) ** 2
    masses = jnp.linspace(0.31, 1.19, 173)
    s = masses**2
    real = jnp.asarray((1.0, -0.4, 1.7, 0.2, 1.1))
    imaginary = jnp.asarray((0.2, 0.8, -0.6, 1.2, -0.1))

    index = jnp.clip(
        jnp.searchsorted(knot_s, s, side="right") - 1,
        0,
        len(knots) - 2,
    )
    x0 = knot_s[index]
    x1 = knot_s[index + 1]
    fraction = (s - x0) / (x1 - x0)
    order = jnp.argsort(index).astype(jnp.int32)
    counts = jnp.bincount(index.astype(jnp.int32), length=len(knots) - 1)
    ends = jnp.cumsum(counts).astype(jnp.int32)
    starts = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), ends[:-1]))
    fixed = 0.7 + 0.3j + 0.1 * jnp.sin(s)

    def prepared_objective(real_values, imaginary_values):
        amplitude = _linear_cartesian_qmi_prepared(
            real_values,
            imaginary_values,
            index,
            fraction,
            order,
            starts,
            ends,
        )
        total = fixed + amplitude
        return -jnp.sum(jnp.log(jnp.abs(total) ** 2)) + 0.03 * jnp.sum(
            jnp.abs(amplitude) ** 2
        )

    def reference_objective(real_values, imaginary_values):
        real_interp = real_values[index] + fraction * (
            real_values[index + 1] - real_values[index]
        )
        imaginary_interp = imaginary_values[index] + fraction * (
            imaginary_values[index + 1] - imaginary_values[index]
        )
        amplitude = real_interp + 1j * imaginary_interp
        total = fixed + amplitude
        return -jnp.sum(jnp.log(jnp.abs(total) ** 2)) + 0.03 * jnp.sum(
            jnp.abs(amplitude) ** 2
        )

    prepared_value = prepared_objective(real, imaginary)
    reference_value = reference_objective(real, imaginary)
    prepared_gradient = jax.grad(prepared_objective, argnums=(0, 1))(real, imaginary)
    reference_gradient = jax.grad(reference_objective, argnums=(0, 1))(real, imaginary)

    assert jnp.allclose(prepared_value, reference_value, rtol=1e-13, atol=1e-13)
    assert jnp.allclose(
        prepared_gradient[0],
        reference_gradient[0],
        rtol=2e-11,
        atol=2e-11,
    )
    assert jnp.allclose(
        prepared_gradient[1],
        reference_gradient[1],
        rtol=2e-11,
        atol=2e-11,
    )


def test_linear_cartesian_qmi_inserting_interpolated_knot_preserves_amplitude():
    knots = (0.30, 0.60, 0.90, 1.20)
    real = jnp.asarray((1.0, -0.5, 0.3, 2.0))
    imaginary = jnp.asarray((0.2, 1.5, -0.7, 0.0))
    inserted_mass = 1.05

    old_knot_s = jnp.asarray(knots) ** 2
    inserted_s = inserted_mass**2
    fraction = (inserted_s - old_knot_s[2]) / (old_knot_s[3] - old_knot_s[2])
    inserted_real = real[2] + fraction * (real[3] - real[2])
    inserted_imaginary = imaginary[2] + fraction * (
        imaginary[3] - imaginary[2]
    )

    refined_knots = (0.30, 0.60, 0.90, inserted_mass, 1.20)
    refined_real = (real[0], real[1], real[2], inserted_real, real[3])
    refined_imaginary = (
        imaginary[0],
        imaginary[1],
        imaginary[2],
        inserted_imaginary,
        imaginary[3],
    )

    original = QMI(
        knots=knots,
        real_parts=tuple(real),
        imaginary_parts=tuple(imaginary),
        interpolation="linear",
    )
    refined = QMI(
        knots=refined_knots,
        real_parts=refined_real,
        imaginary_parts=refined_imaginary,
        interpolation="linear",
    )
    masses = jnp.linspace(0.25, 1.25, 1001)

    assert jnp.allclose(
        original(masses, _context()),
        refined(masses, _context()),
        rtol=0.0,
        atol=2e-13,
    )


def test_prepared_linear_cartesian_qmi_inserting_interpolated_knot_preserves_amplitude():
    knots = (0.30, 0.60, 0.90, 1.20)
    real = jnp.asarray((1.0, -0.5, 0.3, 2.0))
    imaginary = jnp.asarray((0.2, 1.5, -0.7, 0.0))
    inserted_mass = 1.05

    old_knot_s = jnp.asarray(knots) ** 2
    inserted_s = inserted_mass**2
    fraction = (inserted_s - old_knot_s[2]) / (old_knot_s[3] - old_knot_s[2])
    inserted_real = real[2] + fraction * (real[3] - real[2])
    inserted_imaginary = imaginary[2] + fraction * (
        imaginary[3] - imaginary[2]
    )

    original = QMI(
        knots=knots,
        real_parts=tuple(real),
        imaginary_parts=tuple(imaginary),
        interpolation="linear",
    )
    refined = QMI(
        knots=(0.30, 0.60, 0.90, inserted_mass, 1.20),
        real_parts=(real[0], real[1], real[2], inserted_real, real[3]),
        imaginary_parts=(
            imaginary[0],
            imaginary[1],
            imaginary[2],
            inserted_imaginary,
            imaginary[3],
        ),
        interpolation="linear",
    )
    masses = jnp.linspace(0.30, 1.20, 1001)
    context = _context()

    original_prepared = original.prepare_mass(masses, context)
    refined_prepared = refined.prepare_mass(masses, context)
    original_values = original.evaluate_prepared(None, original_prepared, context)
    refined_values = refined.evaluate_prepared(None, refined_prepared, context)

    assert jnp.allclose(
        original_values,
        refined_values,
        rtol=0.0,
        atol=2e-13,
    )


def test_cartesian_qmi_dynamic_cache_matches_direct_model():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    owner = "S_QMI"
    knots = (0.30, 0.55, 0.85, 1.20, 1.60)

    real_parts = tuple(
        Parameter.dynamics(
            f"{owner}.real[{i}]",
            value,
            owner=owner,
        )
        for i, value in enumerate((0.8, -0.3, 0.5, 0.2, -0.1))
    )
    imaginary_parts = tuple(
        Parameter.dynamics(
            f"{owner}.imag[{i}]",
            value,
            owner=owner,
        )
        for i, value in enumerate((0.1, 0.6, -0.2, 0.4, 0.3))
    )
    qmi = QMI(
        knots=knots,
        real_parts=real_parts,
        imaginary_parts=imaginary_parts,
        interpolation="linear",
    )
    model = DecayModel(
        channel,
        [
            NonResonant(
                RealImag(0.35, -0.17),
                name="NR",
                normalize_component=False,
            ),
            Resonance(
                owner,
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=1.0,
                width=0.0,
                spin=0,
                lineshape=qmi,
                normalize_component=False,
            ),
        ],
        normalize_components=False,
        normalization_resolution=30,
    )

    data = model.generate_phase_space(128, seed=417)
    normalization = model.generate_phase_space(512, seed=418)
    cache = model.prepare_cache(data, normalization)

    values = {
        f"{owner}.real[0]": 0.72,
        f"{owner}.real[1]": -0.24,
        f"{owner}.real[2]": 0.43,
        f"{owner}.real[3]": 0.11,
        f"{owner}.real[4]": -0.08,
        f"{owner}.imag[0]": 0.16,
        f"{owner}.imag[1]": 0.51,
        f"{owner}.imag[2]": -0.13,
        f"{owner}.imag[3]": 0.34,
        f"{owner}.imag[4]": 0.27,
    }

    cached_intensity, cached_normalization = cache.evaluate(values)
    direct_intensity = model.intensity(data.as_dict(), values)
    direct_normalization = jnp.mean(
        normalization.weights
        * model.intensity(normalization.as_dict(), values)
    )

    assert jnp.allclose(
        cached_intensity,
        direct_intensity,
        rtol=2e-12,
        atol=2e-12,
    )
    assert jnp.allclose(
        cached_normalization,
        direct_normalization,
        rtol=2e-12,
        atol=2e-12,
    )
