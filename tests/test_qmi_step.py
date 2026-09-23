"""Constant-bin QMI boundaries and prepared differentiation."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import QMI, ResonanceContext


def context():
    return ResonanceContext(
        parent_mass=1.96834,
        daughter_masses=(0.13957, 0.13957),
        bachelor_mass=0.13957,
        spin=0,
        pole_mass=1.0,
        pole_width=0.0,
    )


@pytest.mark.parametrize("cartesian", [False, True])
def test_step_boundaries_prepared_gradients_and_hessian(cartesian):
    # One empty bin, unsorted events, repeated boundaries and endpoint clamping.
    masses = jnp.array([1.5, 0.3, 0.5, 0.49, 1.0, 0.5, 0.1, 2.0])
    indices = np.array([3, 0, 1, 0, 3, 1, 0, 3])
    first = jnp.array([1.0, 2.0, 3.0, 4.0])
    second = jnp.array([0.2, -0.1, 0.4, 0.7])

    def model(a, b):
        kwargs = (
            dict(real_parts=tuple(a), imaginary_parts=tuple(b))
            if cartesian
            else dict(magnitudes=tuple(a), phases=tuple(b))
        )
        return QMI(knots=(0.3, 0.5, 0.8, 1.0, 1.5), interpolation="none", **kwargs)

    qmi = model(first, second)
    prepared = qmi.prepare_mass(masses, context())
    expected = (first + 1j * second) if cartesian else first * jnp.exp(1j * second)
    np.testing.assert_allclose(qmi(masses, context()), np.asarray(expected)[indices])
    np.testing.assert_allclose(
        qmi.evaluate_prepared(None, prepared, context()), np.asarray(expected)[indices]
    )

    def loss(a, b, cached):
        q = model(a, b)
        amp = (
            q.evaluate_prepared(None, prepared, context())
            if cached
            else q(masses, context())
        )
        return jnp.sum(jnp.abs(amp - (0.3 + 0.8j)) ** 2)

    for arg in (0, 1):
        np.testing.assert_allclose(
            jax.jit(jax.grad(lambda a, b: loss(a, b, True), arg))(first, second),
            jax.grad(lambda a, b: loss(a, b, False), arg)(first, second),
            atol=1e-12,
        )
    np.testing.assert_allclose(
        jax.hessian(lambda a: loss(a, second, True))(first),
        jax.hessian(lambda a: loss(a, second, False))(first),
        atol=1e-12,
    )
    assert float(qmi.smoothness_constraint(strength=0)({})) == 0
    with pytest.raises(ValueError, match="constant bins"):
        qmi.smoothness_constraint(strength=1)


def test_step_single_bin_and_lengths():
    q = QMI(knots=(0.3, 1.0), magnitudes=(2.0,), phases=(0.5,), interpolation="none")
    np.testing.assert_allclose(
        q(jnp.array([0.1, 0.3, 0.6, 1.0, 2.0]), context()), 2 * np.exp(0.5j)
    )
    with pytest.raises(ValueError, match="one entry per bin"):
        replace(q, magnitudes=(1.0, 2.0))
