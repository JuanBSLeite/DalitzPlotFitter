import numpy as np
import jax.numpy as jnp
import pytest

from dalitzplotfitter import (
    CPFitSession,
    CPRealImag,
    DecayChannel,
    DecayModel,
    FitSession,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    RealImag,
    Resonance,
    delta_method_covariance,
    delta_method_errors,
    enable_x64,
)

enable_x64()


class FakeCovariance:
    """Minimal stand-in for iminuit's ``Matrix``: name-pair indexable via
    ``covariance[a, b]`` *and* convertible to a plain array via
    ``np.asarray(covariance)``, in whatever order ``names`` declares -- which
    need not match the order any particular caller requests.
    """

    def __init__(self, names, matrix):
        self._names = list(names)
        self._matrix = np.asarray(matrix, dtype=float)

    def __getitem__(self, key):
        a, b = key
        return float(self._matrix[self._names.index(a), self._names.index(b)])

    def __array__(self, dtype=None):
        return np.asarray(self._matrix, dtype=dtype)


def _finite_difference_jacobian(function, values, names):
    base = dict(values)
    jacobian = None
    for j, name in enumerate(names):
        step = 1e-6 * max(1.0, abs(base[name]))
        hi, lo = dict(base), dict(base)
        hi[name] = base[name] + step
        lo[name] = base[name] - step
        difference = np.asarray(function(hi)) - np.asarray(function(lo))
        if jacobian is None:
            jacobian = np.zeros((difference.size, len(names)))
        jacobian[:, j] = difference / (2.0 * step)
    return jacobian


def test_delta_method_errors_matches_hand_derived_variance():
    # f(a, b) = [a**2 + b, a - 2*b]; Cov(a, b) known analytically below.
    def f(values):
        a, b = values["a"], values["b"]
        return jnp.asarray([a**2 + b, a - 2.0 * b])

    values = {"a": 3.0, "b": 1.0}
    names = ["a", "b"]
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])

    errors = np.asarray(delta_method_errors(f, values, names, covariance))

    jacobian = np.array([[2.0 * values["a"], 1.0], [1.0, -2.0]])
    expected_variance = np.diag(jacobian @ covariance @ jacobian.T)
    np.testing.assert_allclose(errors, np.sqrt(expected_variance), rtol=1e-10)


def test_delta_method_covariance_prefers_name_indexing_over_matching_shape():
    """Regression test for a real bug: a shape-based "is this covariance
    already a dense matrix ordered like `names`?" heuristic silently accepted
    an iminuit-style covariance in its own internal order whenever the number
    of requested `names` happened to equal the raw matrix's own dimension --
    exactly the common case of requesting every free parameter. Name
    indexing must always be tried first, regardless of shape.
    """
    def f(values):
        return jnp.asarray([2.0 * values["a"], 3.0 * values["b"]])

    values = {"a": 1.0, "b": 1.0}
    names = ["a", "b"]
    # Declared/stored internally in the OPPOSITE order from `names` -- both
    # array-convertible (in its own order) and name-pair indexable (the
    # semantically correct lookup), exactly like iminuit's Matrix.
    covariance = FakeCovariance(["b", "a"], [[0.09, 0.0], [0.0, 0.01]])

    result = np.asarray(delta_method_covariance(f, values, names, covariance))
    # Correct (name-indexed): Var(a)=0.01, Var(b)=0.09 -> Var(2a)=0.04, Var(3b)=0.81.
    # A shape-based shortcut would instead read the matrix positionally as if
    # already ordered [a, b], giving [0.36, 0.09] instead.
    np.testing.assert_allclose(np.diag(result), [0.04, 0.81], rtol=1e-10)


def _swave_model():
    x_r = Parameter.coefficient("rho.x", 1.0, owner="rho", step=0.01)
    y_r = Parameter.coefficient("rho.y", 0.2, owner="rho", step=0.01)
    x_nr = Parameter.coefficient("NR.x", 0.4, owner="NR", step=0.01)
    y_nr = Parameter.coefficient("NR.y", -0.3, owner="NR", step=0.01)
    return DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [
            Resonance("rho", (0, 1), RealImag(x_r, y_r), mass=0.7753, width=0.1491, spin=1),
            NonResonant(RealImag(x_nr, y_nr), name="NR"),
        ],
        normalization_method="square-dalitz",
        normalization_pair=(0, 1),
        normalization_resolution=24,
    )


def test_decay_model_fit_fraction_errors_matches_finite_differences():
    model = _swave_model()
    values = {parameter.name: parameter.value for parameter in model.parameters}
    names = [parameter.name for parameter in model.parameters]
    rng = np.random.default_rng(0)
    jitter = rng.normal(scale=0.02, size=(len(names), len(names)))
    covariance = 0.01 * np.eye(len(names)) + 0.001 * (jitter + jitter.T)

    errors = model.fit_fraction_errors(values, FakeCovariance(names, covariance))

    def ff(vals):
        return np.asarray(model.fit_fractions(vals))

    jacobian = _finite_difference_jacobian(ff, values, names)
    expected_variance = np.diag(jacobian @ covariance @ jacobian.T)
    component_names = [component.name for component in model.amplitude_model.components]
    for name, expected in zip(component_names, np.sqrt(np.clip(expected_variance, 0.0, None))):
        assert errors[name] == pytest.approx(expected, rel=1e-4)


def test_fit_session_fit_fraction_errors_delegates_to_model():
    model = _swave_model()
    data = PhaseSpaceSample(
        s12=jnp.asarray([0.2, 0.3]),
        s13=jnp.asarray([0.4, 0.5]),
        s23=jnp.asarray([2.0, 1.8]),
        weights=jnp.ones(2),
    )
    session = FitSession(model, data)
    values = {parameter.name: parameter.value for parameter in model.parameters}
    names = [parameter.name for parameter in model.parameters]
    covariance = FakeCovariance(names, 0.01 * np.eye(len(names)))

    class FakeResult:
        def __init__(self, values, covariance):
            self.values = values
            self.covariance = covariance

    errors = session.fit_fraction_errors(FakeResult(values, covariance))
    assert set(errors) == {"rho", "NR"}
    assert all(error >= 0.0 for error in errors.values())


def _cp_swave_models():
    x_r = Parameter.coefficient("rho.x", 1.0, owner="rho", step=0.01)
    y_r = Parameter.coefficient("rho.y", 0.1, owner="rho", step=0.01)
    dx_r = Parameter.coefficient("rho.dx", 0.05, owner="rho", step=0.01)
    dy_r = Parameter.coefficient("rho.dy", -0.02, owner="rho", step=0.01)
    cp = CPRealImag(x_r, y_r, dx_r, dy_r)
    x_nr = Parameter.coefficient("NR.x", 0.4, owner="NR", fixed=True)
    y_nr = Parameter.coefficient("NR.y", -0.1, owner="NR", fixed=True)

    def components(charge):
        return [
            Resonance("rho", (0, 1), cp.for_charge(charge), mass=0.7753, width=0.1491, spin=1),
            NonResonant(RealImag(x_nr, y_nr), name="NR"),
        ]

    plus = DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")), components(+1),
        normalization_method="square-dalitz", normalization_pair=(0, 1), normalization_resolution=24,
    )
    minus = DecayModel(
        DecayChannel("D-", ("pi+", "pi-", "pi-")), components(-1),
        normalization_method="square-dalitz", normalization_pair=(0, 1), normalization_resolution=24,
    )
    return plus, minus


def test_cp_fit_session_fit_fraction_errors_uses_joint_covariance():
    """Regression test: with exactly as many free parameters as the raw
    covariance matrix's own dimension, a shape-based "is this already a
    dense ordered matrix" heuristic would silently accept the covariance in
    its own (wrong) internal order instead of `parameter_names`'s order,
    corrupting cross-charge covariance while leaving each charge's own
    variance untouched (since v^T C v is insensitive to a self-paired
    permutation). Exactly 4 free parameters here (rho.x/.y/.dx/.dy)
    reproduces that exact collision.
    """
    plus, minus = _cp_swave_models()
    plus_data = PhaseSpaceSample(
        s12=jnp.asarray([0.2, 0.3]), s13=jnp.asarray([0.4, 0.5]), s23=jnp.asarray([2.0, 1.8]),
        weights=jnp.ones(2),
    )
    minus_data = PhaseSpaceSample(
        s12=jnp.asarray([0.25, 0.32]), s13=jnp.asarray([0.42, 0.52]), s23=jnp.asarray([1.9, 1.7]),
        weights=jnp.ones(2),
    )
    session = CPFitSession(plus, minus, plus_data, minus_data)
    free_names = [p.name for p in session.parameters if not p.fixed]
    assert len(free_names) == 4  # the exact collision size described above

    values = {p.name: p.value for p in session.parameters}
    rng = np.random.default_rng(1)
    jitter = rng.normal(scale=0.02, size=(len(free_names), len(free_names)))
    covariance_matrix = 0.02 * np.eye(len(free_names)) + 0.003 * (jitter + jitter.T)
    covariance = FakeCovariance(free_names, covariance_matrix)

    class FakeResult:
        def __init__(self, values, covariance):
            self.values = values
            self.covariance = covariance

    errors = session.fit_fraction_errors(FakeResult(values, covariance))

    plus_cache = plus._fraction_cache(None, None)
    minus_cache = minus._fraction_cache(None, None)
    component_names = [c.name for c in plus_cache.components]

    def joint_ff(vals):
        return np.concatenate([
            np.asarray(plus_cache.fit_fractions(vals)),
            np.asarray(minus_cache.fit_fractions(vals)),
        ])

    n = len(component_names)
    jacobian = _finite_difference_jacobian(joint_ff, values, free_names)
    full_covariance = jacobian @ covariance_matrix @ jacobian.T
    variance_plus = np.clip(np.diag(full_covariance)[:n], 0.0, None)
    variance_minus = np.clip(np.diag(full_covariance)[n:], 0.0, None)
    cross = np.diag(full_covariance[:n, n:])
    variance_mean = np.clip(0.25 * (variance_plus + variance_minus + 2.0 * cross), 0.0, None)

    for i, name in enumerate(component_names):
        assert errors["plus"][name] == pytest.approx(np.sqrt(variance_plus[i]), rel=1e-4)
        assert errors["minus"][name] == pytest.approx(np.sqrt(variance_minus[i]), rel=1e-4)
        assert errors["mean"][name] == pytest.approx(np.sqrt(variance_mean[i]), rel=1e-4)
