import jax
import jax.numpy as jnp
import pytest

from dalitzplotfitter import enable_x64
from dalitzplotfitter.likelihood import WeightedUnbinnedNLL

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
