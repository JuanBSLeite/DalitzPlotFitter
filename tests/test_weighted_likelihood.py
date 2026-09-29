import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpwa import enable_x64
from jaxpwa.likelihood import WeightedUnbinnedNLL
from jaxpwa.likelihood.weighted import (
    sandwich_covariance_from_score_outer,
    sweight_covariance_from_hessians,
)

enable_x64()


def _scaled_logpdf(data, parameters):
    return -0.5 * parameters["scale"] * data["x"] ** 2


def test_weighted_unbinned_nll_matches_manual_sum():
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    weights = jnp.array([1.0, 0.5, -0.2])
    nll = WeightedUnbinnedNLL(_scaled_logpdf, data, weights)
    parameters = {"scale": 2.0}
    expected = -jnp.sum(weights * _scaled_logpdf(data, parameters))
    assert jnp.allclose(nll(parameters), expected, rtol=1e-12, atol=1e-12)


def test_weighted_unbinned_nll_is_differentiable():
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    weights = jnp.array([1.0, 0.5, 0.2])
    nll = WeightedUnbinnedNLL(_scaled_logpdf, data, weights)
    grad = jax.grad(lambda parameters: nll(parameters))({"scale": 2.0})
    expected = -jnp.sum(weights * (-0.5 * data["x"] ** 2))
    assert jnp.allclose(grad["scale"], expected, rtol=1e-12, atol=1e-12)


def test_weighted_unbinned_nll_rejects_empty_data():
    with pytest.raises(ValueError, match="non-empty"):
        WeightedUnbinnedNLL(_scaled_logpdf, {}, jnp.array([]))


def test_weighted_unbinned_nll_rejects_mismatched_weight_shape():
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    with pytest.raises(ValueError, match="weights must have shape"):
        WeightedUnbinnedNLL(_scaled_logpdf, data, jnp.array([1.0, 1.0]))


def test_weighted_unbinned_nll_rejects_non_finite_weights():
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    with pytest.raises(ValueError, match="finite"):
        WeightedUnbinnedNLL(_scaled_logpdf, data, jnp.array([1.0, jnp.nan, 1.0]))


def test_weighted_unbinned_nll_rejects_logpdf_shape_mismatch():
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    weights = jnp.array([1.0, 1.0, 1.0])
    nll = WeightedUnbinnedNLL(lambda d, p: jnp.array([0.0, 0.0]), data, weights)
    with pytest.raises(ValueError, match="logpdf must return shape"):
        nll({})


def test_sweight_covariance_from_hessians_matches_matrix_formula():
    weighted_hessian = np.array([[5.0, 1.0], [1.0, 3.0]])
    squared_weight_hessian = np.array([[7.0, 0.5], [0.5, 4.0]])
    inverse = np.linalg.inv(weighted_hessian)
    expected = inverse @ squared_weight_hessian @ inverse
    actual = sweight_covariance_from_hessians(
        weighted_hessian,
        squared_weight_hessian,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_sandwich_covariance_from_score_outer_matches_matrix_formula():
    weighted_hessian = np.array([[5.0, 1.0], [1.0, 3.0]])
    score_outer = np.array([[8.0, 1.5], [1.5, 6.0]])
    inverse = np.linalg.inv(weighted_hessian)
    expected = inverse @ score_outer @ inverse
    actual = sandwich_covariance_from_score_outer(weighted_hessian, score_outer)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_sweight_covariance_from_hessians_rejects_bad_shapes():
    with pytest.raises(ValueError, match="square"):
        sweight_covariance_from_hessians(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="same shape"):
        sweight_covariance_from_hessians(np.eye(2), np.eye(3))


def test_sweight_covariance_from_hessians_rejects_singular_hessian():
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        sweight_covariance_from_hessians(
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.eye(2),
        )
