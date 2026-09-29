"""Numerical regression checks for the notebook-local regularized sandwich.

Load only the objective/helper definitions; never execute ROOT I/O or data fits.
The reference calculation differentiates explicit event losses independently.
"""

import ast
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import Minimizer, Parameter, WeightedUnbinnedNLL, enable_x64
from jaxpwa.likelihood.weighted import sandwich_covariance_from_score_outer
from jaxpwa.workflow import _install_minuit_covariance

enable_x64()


@pytest.fixture(scope="module")
def notebook_helpers():
    path = Path(__file__).resolve().parents[1] / (
        "notebooks/examples/"
        "ds_pipipi_lhcb2023_goofit_real_data_fit_cow_sweights_logbarrier.ipynb"
    )
    notebook = json.loads(path.read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "class NegativeWeightLogBarrierNLL:" in "".join(cell["source"])
    )
    namespace = {
        "dataclass": dataclass, "jax": jax, "jnp": jnp, "np": np,
        "warnings": warnings, "Minimizer": Minimizer,
        "WeightedUnbinnedNLL": WeightedUnbinnedNLL,
        "sandwich_covariance_from_score_outer": sandwich_covariance_from_score_outer,
        "_install_minuit_covariance": _install_minuit_covariance,
    }
    exec(compile(source, str(path), "exec"), namespace)
    fit_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def fit_barrier(" in "".join(cell["source"])
    )
    fit_function = next(
        node for node in ast.parse(fit_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "fit_barrier"
    )
    return SimpleNamespace(
        objective=namespace["NegativeWeightLogBarrierNLL"],
        correct=namespace["apply_barrier_sandwich"],
        namespace=namespace,
        fit_function=fit_function,
    )


def _normal_logpdf(data, parameters):
    scale = jnp.exp(parameters.get("log_scale", 0.0))
    residual = (data["x"] - parameters["mu"]) / scale
    return -0.5 * residual**2 - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)


def _example(helpers, strength):
    data = {"x": jnp.array([-1.0, 0.0, 0.2, 0.8, 1.0, 3.0])}
    weights = jnp.array([1.2, 1.0, 0.8, 1.0, 1.1, -0.4])
    log_reference = _normal_logpdf(data, {"mu": 0.0, "log_scale": jnp.log(2.0)})
    return helpers.objective(
        _normal_logpdf, data, weights, log_reference,
        strength=strength, epsilon=0.5,
    )


def _event_losses(objective, vector):
    """Independent direct event-loss reference, without effective score weights."""
    logp = _normal_logpdf(
        objective.data, {"mu": vector[0], "log_scale": vector[1]},
    )
    penalty = jnp.where(
        objective.weights < 0,
        objective.strength * (-objective.weights) * jnp.maximum(
            jnp.log(objective.epsilon) + objective.log_reference - logp, 0.0,
        )**2,
        0.0,
    )
    return -objective.weights * logp + penalty


@pytest.mark.parametrize("strength", [0.0, 1.3])
def test_regularized_matrices_match_direct_event_derivatives(
    notebook_helpers, strength,
):
    objective = _example(notebook_helpers, strength)
    point = jnp.array([0.1, 0.0])
    parameters = [Parameter.coefficient("mu", 0.1),
                  Parameter.coefficient("log_scale", 0.0)]
    fitted = {"mu": point[0], "log_scale": point[1]}
    names, sensitivity = Minimizer(objective, parameters).jax_hessian(fitted)
    score_objective = objective.score_outer_objective(fitted)
    score_names, variability = Minimizer(
        score_objective, parameters,
    ).jax_hessian(fitted)

    def losses(vector):
        return _event_losses(objective, vector)

    event_gradients = np.asarray(jax.jacrev(losses)(point))
    expected_a = np.asarray(jax.hessian(lambda vector: jnp.sum(losses(vector)))(point))
    expected_b = event_gradients.T @ event_gradients
    assert names == score_names == ("mu", "log_scale")
    np.testing.assert_allclose(sensitivity, expected_a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(variability, expected_b, rtol=1e-12, atol=1e-12)

    # Also compare against a finite-difference event-score outer product.
    step = 1e-5
    finite_scores = np.column_stack([
        (np.asarray(losses(point + step * direction))
         - np.asarray(losses(point - step * direction))) / (2 * step)
        for direction in np.eye(2)
    ])
    np.testing.assert_allclose(variability, finite_scores.T @ finite_scores,
                               rtol=1e-8, atol=1e-8)
    plain = _example(notebook_helpers, 0.0)
    plain_scores = np.asarray(jax.jacrev(lambda v: _event_losses(plain, v))(point))
    if strength:
        assert not np.allclose(variability, plain_scores.T @ plain_scores)
        assert float(objective.terms(fitted)[1]) > 0
    else:
        np.testing.assert_allclose(variability, plain_scores.T @ plain_scores)


@pytest.mark.parametrize("strength", [0.0, 1.3])
def test_corrected_errors_installed_after_actual_fit(notebook_helpers, strength):
    objective = _example(notebook_helpers, strength)
    parameters = [Parameter.coefficient("mu", 0.1, step=0.05)]
    minimizer = Minimizer(objective, parameters, tolerance=1e-6)
    result = minimizer.fit(ncall=1000, hesse=False)
    assert result.valid
    optimizer_matrix = np.array(result.covariance, copy=True)
    optimum = float(result.values["mu"])
    report = notebook_helpers.correct(objective, minimizer, result)
    assert report["status"] == "applied"
    point = jnp.array([optimum, 0.0])
    def losses(mu):
        return _event_losses(objective, point.at[0].set(mu))

    scores = np.asarray(jax.jacrev(losses)(optimum))
    curvature = float(jax.hessian(lambda mu: jnp.sum(losses(mu)))(optimum))
    expected_variance = np.sum(scores**2) / curvature**2
    assert report["names"] == ("mu",)
    np.testing.assert_allclose(result.covariance["mu", "mu"], expected_variance,
                               rtol=1e-12)
    np.testing.assert_allclose(result.errors["mu"], np.sqrt(expected_variance),
                               rtol=1e-12)
    np.testing.assert_array_equal(report["optimizer_covariance"], optimizer_matrix)
    assert result.values["mu"] == optimum
    if strength:
        assert float(objective.terms({"mu": optimum})[1]) > 0


def test_notebook_fit_wrapper_corrects_each_fit_from_shared_start(notebook_helpers):
    scope = dict(notebook_helpers.namespace)
    shared_start = {"mu": 0.1}
    scope.update(
        session=SimpleNamespace(parameters=[Parameter.coefficient("mu", 0.1)]),
        dp_start=shared_start, FIT_TOLERANCE=1e-6, FIT_STRATEGY=1,
        FIT_NCALL=1000, FIT_VERBOSE=0, COMPUTE_BARRIER_COVARIANCE=True,
        COVARIANCE_HESSIAN_BATCH_SIZE=1, barrier_covariance_reports={},
    )
    module = ast.Module(body=[notebook_helpers.fit_function], type_ignores=[])
    exec(compile(module, "<notebook fit_barrier>", "exec"), scope)
    results = [
        scope["fit_barrier"](_example(notebook_helpers, strength))
        for strength in (0.0, 1.3)
    ]
    for result in results:
        report = scope["barrier_covariance_reports"][id(result)]
        assert report["status"] == "applied"
        np.testing.assert_allclose(result.errors["mu"], report["errors"]["mu"])
    assert shared_start == {"mu": 0.1}


def test_invalid_fit_skips_correction_without_changing_errors(notebook_helpers):
    result = SimpleNamespace(valid=False, covariance="original", errors="original")
    with pytest.warns(RuntimeWarning, match="Invalid fit"):
        report = notebook_helpers.correct(None, None, result)
    assert report["status"] == "skipped_invalid_fit"
    assert result.covariance == result.errors == "original"


@pytest.mark.parametrize("hessian", [np.array([[0.0]]), np.array([[-1.0]]),
                                     np.array([[np.nan]])])
def test_unusable_hessian_does_not_install_corrected_errors(notebook_helpers, hessian):
    result = SimpleNamespace(valid=True, fval=1., values={"mu": 0.},
                             covariance="original", errors="original")
    minimizer = SimpleNamespace(
        parameters=[Parameter.coefficient("mu", 0.)],
        jax_hessian=lambda values: (("mu",), hessian),
    )
    with pytest.warns(RuntimeWarning, match="sandwich failed"):
        report = notebook_helpers.correct(None, minimizer, result)
    assert report["status"] == "failed"
    assert result.covariance == result.errors == "original"
