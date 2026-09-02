import jax.numpy as jnp

from dalitzplotfitter import ConstrainedNLL, GaussianConstraint, Parameter


def test_gaussian_constraint_matches_quadratic_penalty():
    p = Parameter("x", 0.0)
    constraint = GaussianConstraint(p, mean=1.0, sigma=0.5)
    assert jnp.isclose(constraint({"x": 2.0}), 2.0)


def test_constrained_nll_adds_penalty():
    p = Parameter("x", 0.0)
    base = lambda pars: jnp.asarray(3.0)
    nll = ConstrainedNLL(base, GaussianConstraint(p, mean=1.0, sigma=0.5))
    assert jnp.isclose(nll({"x": 2.0}), 5.0)
