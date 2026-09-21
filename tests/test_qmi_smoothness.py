"""Discrete QMI regularization, independent of interpolation and event caching."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    QMI,
    ConstrainedNLL,
    CPFitSession,
    DecayChannel,
    DecayModel,
    FitSession,
    Parameter,
    QMISmoothnessConstraint,
    RealImag,
    Resonance,
    ResonanceContext,
)

MODES = ("linear", "cubic", "hermite", "natural")


def _shape(mode="linear", polar=False, prefix="S"):
    knots = (0.3, 0.55, 0.9, 1.4)
    first = (1.0, 0.8, 1.3, 1.1)
    second = (0.2, -0.4, 0.6, 0.9)
    fields = ("magnitudes", "phases") if polar else ("real_parts", "imaginary_parts")
    parts = {
        field: tuple(
            Parameter.dynamics(f"{prefix}.{field}[{i}]", value, owner="S")
            for i, value in enumerate(values)
        )
        for field, values in zip(fields, (first, second), strict=True)
    }
    return QMI(knots, interpolation=mode, **parts)


def _groups(qmi):
    if qmi.parameterization == "polar":
        return qmi.magnitudes, qmi.phases
    return qmi.real_parts, qmi.imaginary_parts


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("polar", (False, True))
def test_modes_parameter_resolution_jit_gradient_and_hessian(mode, polar):
    shape = _shape(mode, polar)
    params = [p for group in _groups(shape) for p in group]
    theta = jnp.array([p.value for p in params])
    constraint = shape.smoothness_constraint(0.7, weights=(0.4, 1.2))

    def objective(vector):
        return constraint(dict(zip([p.name for p in params], vector, strict=True)))

    # JIT gradients and Hessians must work for both node coordinate systems.
    value, gradient = jax.jit(jax.value_and_grad(objective))(theta)
    assert float(value) == pytest.approx(float(constraint({})))
    hessian = np.asarray(jax.jit(jax.hessian(objective))(theta))
    assert np.isfinite(hessian).all()
    np.testing.assert_allclose(hessian, hessian.T, rtol=1e-10, atol=1e-10)
    step = 1e-5
    fd_gradient = []
    fd_hessian = []
    for direction in np.eye(len(theta)):
        fd_gradient.append(
            (objective(theta + step * direction) - objective(theta - step * direction))
            / (2 * step)
        )
        fd_hessian.append(
            (
                jax.grad(objective)(theta + step * direction)
                - jax.grad(objective)(theta - step * direction)
            )
            / (2 * step)
        )
    np.testing.assert_allclose(gradient, fd_gradient, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(hessian, np.array(fd_hessian).T, rtol=2e-8, atol=2e-8)
    reference = replace(shape, interpolation="linear").smoothness_constraint(
        0.7, weights=(0.4, 1.2)
    )
    assert float(value) == pytest.approx(float(reference({})))
    if not polar:
        assert np.linalg.eigvalsh(hessian).min() > -1e-9


@pytest.mark.parametrize("mode", MODES)
def test_exact_curvature_for_quadratic_on_nonuniform_mass_squared_grid(mode):
    # q(s) = (1+2i)s². |q''|² = 20, interior dual-cell width = (4-1)/2.
    s = np.array([1.0, 2.0, 4.0])
    z = (1 + 2j) * s**2
    shape = QMI(
        tuple(np.sqrt(s)),
        real_parts=tuple(z.real),
        imaginary_parts=tuple(z.imag),
        interpolation=mode,
    )
    assert float(shape.smoothness_constraint(1.0)({})) == pytest.approx(30.0)
    assert float(shape.smoothness_constraint(0.3, weights=(2.0,))({})) == pytest.approx(
        18.0
    )
    polar = QMI(
        shape.knots,
        magnitudes=tuple(abs(z)),
        phases=tuple(np.angle(z)),
        interpolation=mode,
    )
    assert float(polar.smoothness_constraint(1.0)({})) == pytest.approx(30.0)


@pytest.mark.parametrize("mode", MODES)
def test_affine_complex_nodes_are_unpenalized_even_for_nonlinear_interpolants(mode):
    s = np.array([0.09, 0.2, 0.7, 2.0])
    z = (1 + 0.2j) + (0.3 - 0.7j) * s
    qmi = QMI(
        tuple(np.sqrt(s)),
        real_parts=tuple(z.real),
        imaginary_parts=tuple(z.imag),
        interpolation=mode,
    )
    assert float(qmi.smoothness_constraint(2.0)({})) < 1e-27


def test_phase_rotation_and_polar_wrap_do_not_change_penalty():
    qmi = QMI(
        (0.3, 0.6, 0.9, 1.5), magnitudes=(1, 2, 0.1, 1.3), phases=(3.1, -3.1, 0.2, 0.8)
    )
    expected = float(qmi.smoothness_constraint(0.2)({}))
    rotated = replace(qmi, phases=tuple(p + 1.1 for p in qmi.phases))
    wrapped = replace(
        qmi,
        phases=tuple(
            p + k * 2 * np.pi for p, k in zip(qmi.phases, (2, -1, 0, 3), strict=True)
        ),
    )
    assert float(rotated.smoothness_constraint(0.2)({})) == pytest.approx(expected)
    assert float(wrapped.smoothness_constraint(0.2)({})) == pytest.approx(expected)


def test_zero_strength_disabled_stencils_two_knots_and_zero_amplitude():
    qmi = _shape()
    names = [p.name for group in _groups(qmi) for p in group]
    zero = qmi.smoothness_constraint()
    values = {n: jnp.asarray(0.7) for n in names}
    assert float(zero(values)) == 0
    assert all(float(v) == 0 for v in jax.grad(zero)(values).values())
    assert float(qmi.smoothness_constraint(2.0, weights=(0, 0))(values)) == 0
    two = QMI((0.3, 0.6), real_parts=(0.0, 4.0), imaginary_parts=(0.0, 2.0))
    assert float(two.smoothness_constraint(10.0, weights=())({})) == 0
    penalty = qmi.smoothness_constraint(1.0)
    zeros = {n: jnp.asarray(0.0) for n in names}
    assert float(penalty(zeros)) == 0
    assert all(np.isfinite(float(v)) for v in jax.grad(penalty)(zeros).values())


def test_mask_disables_entire_stencil_and_keeps_other_stencils_active():
    qmi = QMI(
        (1.0, np.sqrt(2), np.sqrt(3), 2.0),
        real_parts=(0, 0, 0, 1),
        imaginary_parts=(0, 0, 0, 0),
    )
    assert float(qmi.smoothness_constraint(1.0, weights=(1, 0))({})) == 0
    assert float(qmi.smoothness_constraint(1.0, weights=(0, 1))({})) == pytest.approx(
        1.0
    )


@pytest.mark.parametrize("strength", (-1, np.inf, np.nan, Parameter("lambda", 1)))
def test_invalid_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="strength"):
        _shape().smoothness_constraint(strength)


@pytest.mark.parametrize(
    "weights",
    ((1,), (1, 2, 3), (-1, 1), (np.nan, 1), (1, np.inf), (Parameter("w", 1), 1)),
)
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weight"):
        _shape().smoothness_constraint(1, weights=weights)


def test_requires_qmi_and_freezes_weights():
    with pytest.raises(TypeError, match="1D QMI"):
        QMISmoothnessConstraint(object(), 1)
    weights = [1, 2]
    constraint = _shape().smoothness_constraint(1, weights=weights)
    weights[0] = 0
    assert constraint.weights == (1, 2)


def _model(shape, minus=False):
    channel = DecayChannel(
        "B-" if minus else "B+",
        ("pi-", "pi-", "pi+") if minus else ("pi+", "pi+", "pi-"),
    )
    return DecayModel(
        channel,
        [
            Resonance(
                "S",
                (0, 2),
                RealImag(1.0, 0.0),
                lineshape=shape,
                mass=1.0,
                width=0.1,
                spin=0,
                normalize_component=False,
            )
        ],
        normalization_method="square-dalitz",
        normalization_resolution=8,
        normalization_pair=(0, 1),
    )


@pytest.mark.parametrize("mode", MODES)
def test_session_adds_penalty_once_without_changing_parameters_or_pdf(mode):
    qmi = _shape(mode)
    model = _model(qmi)
    sample = model.generate_phase_space(12, seed=728)
    session = FitSession(model, sample)
    penalty = qmi.smoothness_constraint(0.01)
    constrained = session.with_constraint(penalty)
    values = {p.name: p.value for p in session.parameters}
    assert constrained.parameters == session.parameters
    base = float(session.objective(values))
    assert float(constrained.objective(values)) == pytest.approx(
        base + float(penalty(values))
    )
    assert (
        float(session.with_constraint(qmi.smoothness_constraint()).objective(values))
        == base
    )
    np.testing.assert_array_equal(
        constrained.model.amplitude(sample.as_dict(), values),
        model.amplitude(sample.as_dict(), values),
    )
    # Low-level NLL path gives the same scalar and gradient as the session.
    low_level = ConstrainedNLL(session.base_objective, penalty)
    np.testing.assert_allclose(
        float(low_level(values)), float(constrained.objective(values))
    )
    actual = jax.grad(constrained.objective)(values)
    base_gradient = jax.grad(session.objective)(values)
    penalty_gradient = jax.grad(penalty)(values)
    for name in values:
        assert float(actual[name]) == pytest.approx(
            float(base_gradient[name] + penalty_gradient[name]), rel=1e-9, abs=1e-9
        )


def test_cp_session_keeps_charge_penalties_separate_and_excludes_other_parameters():
    plus_qmi, minus_qmi = _shape(prefix="plus"), _shape(prefix="minus", polar=True)
    plus, minus = _model(plus_qmi), _model(minus_qmi, minus=True)
    session = CPFitSession(
        plus,
        minus,
        plus.generate_phase_space(8, seed=19),
        minus.generate_phase_space(9, seed=20),
    )
    plus_penalty, minus_penalty = (
        plus_qmi.smoothness_constraint(0.01),
        minus_qmi.smoothness_constraint(0.01),
    )
    constrained = session.with_constraint(plus_penalty).with_constraint(minus_penalty)
    assert constrained.parameters == session.parameters
    values = {p.name: p.value for p in session.parameters}
    assert float(
        constrained.objective(values) - session.objective(values)
    ) == pytest.approx(float(plus_penalty(values) + minus_penalty(values)))
    gradients = jax.grad(plus_penalty)({**values, "chi_c0.x": 0.4})
    assert gradients["chi_c0.x"] == 0
    assert all(float(v) == 0 for n, v in gradients.items() if n.startswith("minus"))
    assert any(abs(float(v)) > 0 for n, v in gradients.items() if n.startswith("plus"))


def test_constraint_creation_does_not_change_prepared_interpolator():
    shape = QMI(
        (0.3, 0.6, 1.0),
        real_parts=(1, 2, 1),
        imaginary_parts=(0, 1, 0),
        interpolation="natural",
    )
    context = ResonanceContext(
        parent_mass=5.279,
        daughter_masses=(0.139, 0.139),
        bachelor_mass=0.139,
        spin=0,
        pole_mass=1.0,
        pole_width=0.1,
    )
    masses = jnp.linspace(0.3, 1.0, 15)
    before = shape(masses, context)
    penalty = shape.smoothness_constraint(10)
    assert float(penalty({})) > 0
    np.testing.assert_array_equal(before, shape(masses, context))
