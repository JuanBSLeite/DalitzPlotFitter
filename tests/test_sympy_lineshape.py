"""Optional symbolic dynamics: numerical equivalence and fit integration."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    NonResonant,
    Parameter,
    Pole,
    RealImag,
    Resonance,
    ResonanceContext,
    SympyLineshape,
)

sp = pytest.importorskip("sympy")
m, m0, width, alpha = sp.symbols("m m0 width alpha", real=True)
CONTEXT = ResonanceContext(1.869, (0.139, 0.139), 0.139, 0, 0.77, 0.15)


def test_optional_dependency_is_lazy():
    code = """
import importlib.abc
import sys
class NoSympy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sympy" or fullname.startswith("sympy."):
            raise ImportError("blocked for optional dependency test")
sys.meta_path.insert(0, NoSympy())
import dalitzplotfitter as dpf
assert "sympy" not in sys.modules
assert isinstance(dpf.Resonance("x", (0, 1), 1).lineshape, dpf.RelativisticBreitWigner)
try:
    dpf.SympyLineshape(None, mass_symbol=None)
except ImportError as error:
    assert "dalitzplotfitter[sympy]" in str(error)
else:
    raise AssertionError("missing optional dependency was not reported")
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code], check=True, env=os.environ.copy()
    )


def pole_shape():
    return SympyLineshape(
        1 / (m - m0 - sp.I * width / 2),
        mass_symbol=m,
        context_symbols={m0: "pole_mass", width: "pole_width"},
    )


@pytest.mark.parametrize("shape", [(), (7,), (2, 3), (0,)])
def test_pole_equivalence_jit_and_shape(shape):
    mass = jnp.full(shape, 0.8)
    symbolic = pole_shape()
    actual = jax.jit(lambda x: symbolic(x, CONTEXT))(mass)
    np.testing.assert_allclose(actual, Pole()(mass, CONTEXT), rtol=1e-13)
    assert actual.shape == mass.shape
    assert jnp.issubdtype(actual.dtype, jnp.complexfloating)


def test_constant_broadcast_and_immutable_bindings():
    source = {alpha: 2.0}
    shape = SympyLineshape(alpha, mass_symbol=m, parameters=source)
    source[alpha] = 3.0
    shape.parameters[alpha] = 4.0
    np.testing.assert_array_equal(shape(jnp.ones(3), CONTEXT), [2, 2, 2])
    with pytest.raises(FrozenInstanceError):
        shape.expression = sp.Integer(9)


def test_resolution_aliases_fixed_parameters_and_kernel_reuse(monkeypatch):
    p = Parameter.dynamics("shape.alpha", 0.3, owner="shape", backend_name="a")
    shape = SympyLineshape(sp.exp(-alpha * m), mass_symbol=m, parameters={alpha: p})

    def forbidden(*args, **kwargs):
        raise AssertionError("Repeated compilation")

    monkeypatch.setattr(sp, "lambdify", forbidden)
    for values in ({"shape.alpha": 0.4}, {"a": 0.4}):
        resolved = shape.resolve(values)
        assert resolved.kernel is shape._kernel
        np.testing.assert_allclose(resolved(jnp.array([1.0]), CONTEXT), np.exp(-0.4))
    # Same precedence as resolve_value for backend mappings.
    resolved = shape.resolve({"a": 0.4, "shape.alpha": 0.9})
    np.testing.assert_allclose(resolved(jnp.array([1.0]), CONTEXT), np.exp(-0.4))
    gradient = jax.jit(
        jax.grad(lambda a: jnp.real(shape.resolve({"a": a})(jnp.array(1.0), CONTEXT)))
    )(0.4)
    np.testing.assert_allclose(gradient, -np.exp(-0.4))


def test_complex_threshold_and_second_derivative():
    expr = sp.sqrt(1 - alpha**2 / m**2)
    shape = SympyLineshape(
        expr, mass_symbol=m, parameters={alpha: 1.0}, complex_domain=True
    )
    mass = jnp.array([0.8, 1.2])
    result = jax.jit(lambda x: shape(x, CONTEXT))(mass)
    np.testing.assert_allclose(
        result, np.sqrt((1 - 1 / np.asarray(mass) ** 2).astype(complex))
    )
    assert jnp.all(jnp.isfinite(result))
    d2 = jax.grad(jax.grad(lambda x: jnp.real(shape(x, CONTEXT))))(1.2)
    assert jnp.isfinite(d2)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"expression": "m", "mass_symbol": m}, TypeError),
        ({"expression": sp.Matrix([m]), "mass_symbol": m}, TypeError),
        ({"expression": m + alpha, "mass_symbol": m}, ValueError),
        ({"expression": m, "mass_symbol": m, "parameters": {alpha: 1}}, ValueError),
        ({"expression": m, "mass_symbol": "m"}, TypeError),
        ({"expression": m, "mass_symbol": m, "parameters": {m: 1}}, ValueError),
        (
            {
                "expression": alpha,
                "mass_symbol": m,
                "parameters": {alpha: 1},
                "context_symbols": {alpha: "pole_mass"},
            },
            ValueError,
        ),
        (
            {
                "expression": alpha,
                "mass_symbol": m,
                "context_symbols": {alpha: "__class__"},
            },
            ValueError,
        ),
        ({"expression": sp.Function("unknown")(m), "mass_symbol": m}, ValueError),
        (
            {"expression": sp.Piecewise((m, m > 0), (0, True)), "mass_symbol": m},
            ValueError,
        ),
        (
            {
                "expression": alpha * m,
                "mass_symbol": m,
                "parameters": {alpha: Parameter.coefficient("a", 1)},
            },
            ValueError,
        ),
    ],
)
def test_invalid_declarations(kwargs, error):
    with pytest.raises(error):
        SympyLineshape(**kwargs)


def test_json_roundtrip_with_shared_parameter_identity():
    parameter = Parameter.dynamics(
        "shape.a", 0.2, owner="shape", bounds=(0, 1), fixed=True
    )
    shape = SympyLineshape(
        sp.exp(-alpha * m) / (m - m0 - sp.I * width / 2)
        + sp.Float("0.01234567890123456789", 70),
        mass_symbol=m,
        parameters={alpha: parameter, width: 0.15},
        context_symbols={m0: "pole_mass"},
        complex_domain=True,
    )
    spec = json.loads(json.dumps(shape.to_spec()))
    restored = SympyLineshape.from_spec(
        spec, parameter_registry={parameter.name: parameter}
    )
    assert restored.parameters[alpha] is parameter
    mass = jnp.array([0.5, 0.7, 1.1])
    np.testing.assert_allclose(
        restored(mass, CONTEXT), shape(mass, CONTEXT), rtol=1e-14
    )
    assert restored.expression == shape.expression
    conflicting = Parameter.dynamics("shape.a", 0.3, owner="shape", bounds=(0, 1))
    with pytest.raises(ValueError, match="Conflicting"):
        SympyLineshape.from_spec(spec, parameter_registry={parameter.name: conflicting})
    spec["expression"] = {"op": "eval", "args": []}
    with pytest.raises(ValueError, match="Unsupported"):
        SympyLineshape.from_spec(spec)


def make_model(fixed=False, builtin=False):
    p = Parameter.dynamics("shape.mass", 0.77, owner="shape", fixed=fixed)
    w = Parameter.dynamics(
        "shape.width", 0.15, owner="shape", fixed=fixed, backend_name="g"
    )
    shape = SympyLineshape(
        1 / (m - m0 - sp.I * width / 2),
        mass_symbol=m,
        parameters={m0: p, width: w},
    )
    return DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [
            Resonance(
                "shape",
                (0, 1),
                RealImag(0.6, -0.2),
                mass=p,
                width=w,
                spin=1,
                lineshape=Pole() if builtin else shape,
            ),
            NonResonant(RealImag(1.0, 0.0)),
        ],
        normalization_method="square-dalitz",
        normalization_resolution=16,
    )


@pytest.mark.parametrize("fixed", [True, False])
def test_cache_matches_builtin_including_normalization_and_symmetrization(fixed):
    model = make_model(fixed=fixed)
    reference = make_model(fixed=fixed, builtin=True)
    sample = model.normalization_sample
    cache = model.prepare_cache(sample, sample)
    builtin = reference.prepare_cache(sample, sample)
    assert len(model.parameters) == 2  # shared mass/width declarations deduplicate
    assert bool(cache.floating_dynamics) == (not fixed)
    for mass in (0.73, 0.81):
        values = {"shape.mass": mass, "shape.width": 0.18}
        actual = jax.jit(cache.evaluate)(values)
        expected = builtin.evaluate(values)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)


def test_full_normalized_nll_gradient_matches_finite_difference():
    model = make_model()
    sample = model.normalization_sample
    cache = model.prepare_cache(sample, sample)

    def nll(vector):
        values = {"shape.mass": vector[0], "shape.width": vector[1]}
        intensity, normalization = cache.evaluate(values)
        return -jnp.mean(jnp.log(intensity)) + jnp.log(normalization)

    point = jnp.array([0.79, 0.17])
    gradient = jax.jit(jax.grad(nll))(point)
    fd = []
    for i in range(2):
        delta = jnp.zeros(2).at[i].set(1e-6)
        fd.append((nll(point + delta) - nll(point - delta)) / (2e-6))
    np.testing.assert_allclose(gradient, fd, rtol=2e-5, atol=2e-6)
    assert float(jnp.linalg.norm(gradient)) > 1e-4
