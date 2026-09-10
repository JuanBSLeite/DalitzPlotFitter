from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from dalitzplotfitter import QMI


@pytest.mark.parametrize("polar", [False, True])
@pytest.mark.parametrize("knots", [(0.3, 1.8), (0.3, 0.5, 0.9, 1.2, 1.8)])
def test_natural_matches_reference_and_prepared_gradients(polar, knots):
    x = np.square(knots)
    y = np.linspace(0.2, 1.0, len(x)) ** 2
    z = np.sin(np.arange(len(x)))
    mass = jnp.asarray(np.linspace(0.2, 2.0, 41))
    ctx = SimpleNamespace(spin=0)

    def make(values):
        kwargs = (
            dict(magnitudes=values, phases=z)
            if polar
            else dict(real_parts=values, imaginary_parts=z)
        )
        return QMI(knots=knots, interpolation="natural", **kwargs)

    qmi = make(y)
    prepared = qmi.prepare_mass(mass, ctx)
    query = np.clip(np.asarray(mass) ** 2, x[0], x[-1])
    first = CubicSpline(x, y, bc_type="natural")(query)
    second = CubicSpline(x, z, bc_type="natural")(query)
    expected = first * np.exp(1j * second) if polar else first + 1j * second
    np.testing.assert_allclose(qmi(mass, ctx), expected, atol=1e-11)
    np.testing.assert_allclose(
        qmi.evaluate_prepared(None, prepared, ctx), expected, atol=1e-11
    )

    def loss(values, prepared_path):
        q = make(values)
        a = q.evaluate_prepared(None, prepared, ctx) if prepared_path else q(mass, ctx)
        return jnp.sum(jnp.abs(a) ** 2)

    gradient = jax.jit(jax.grad(lambda v: loss(v, True)))(jnp.asarray(y))
    np.testing.assert_allclose(
        gradient, jax.grad(lambda v: loss(v, False))(jnp.asarray(y)), atol=1e-10
    )
    eps = 1e-5
    numerical = [
        (
            float(loss(y + eps * np.eye(len(x))[i], False))
            - float(loss(y - eps * np.eye(len(x))[i], False))
        )
        / (2 * eps)
        for i in range(len(x))
    ]
    np.testing.assert_allclose(gradient, numerical, atol=1e-7)


def test_natural_has_global_support():
    def value(y):
        q = QMI(
            (0.3, 0.5, 0.8, 1.2, 1.8),
            real_parts=y,
            imaginary_parts=(0.0,) * 5,
            interpolation="natural",
        )
        return jnp.real(q(jnp.array([0.4]), SimpleNamespace(spin=0))[0])

    gradient = jax.grad(value)(jnp.ones(5))
    assert abs(float(gradient[-1])) > 1e-6


def test_natural_second_derivatives_and_interior_continuity():
    from dalitzplotfitter.dynamics.lineshape.qmi import _natural_cubic

    x = jnp.array([0.09, 0.25, 0.64, 1.44, 3.24])
    y = jnp.array([0.4, 1.0, -0.2, 0.8, 0.3])

    def segment(s, index):
        return _natural_cubic(y, index, (s - x[index]) / (x[index + 1] - x[index]), x)

    first = jax.grad(segment, argnums=0)
    second = jax.grad(first, argnums=0)
    assert abs(float(second(x[0], 0))) < 1e-10
    assert abs(float(second(x[-1], 3))) < 1e-10
    for i in range(1, 4):
        np.testing.assert_allclose(first(x[i], i - 1), first(x[i], i), atol=1e-10)
        np.testing.assert_allclose(second(x[i], i - 1), second(x[i], i), atol=1e-10)
