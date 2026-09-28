"""Delta-method (linear) error propagation for JAX-differentiable observables."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array


def _covariance_matrix(covariance: object, names: Sequence[str]) -> Array:
    """Build a dense covariance matrix ordered like ``names``.

    ``covariance`` is ordinarily an object supporting name-pair indexing
    ``covariance[a, b]`` -- notably an ``iminuit`` ``Minuit.covariance``
    matrix, which this is designed to accept directly -- and that indexing
    is always tried first. A plain dense array (already ordered like
    ``names``) is accepted as a fallback only when name indexing itself
    fails, never based on matching shape: an ``iminuit`` covariance matrix
    is itself array-like, and if ``names`` happened to list every parameter
    Minuit knows about, a shape-based check would wrongly accept it still in
    Minuit's own internal parameter order, silently pairing the wrong
    covariance entries with the caller's Jacobian columns whenever that
    order differs from ``names``.
    """
    try:
        return jnp.asarray([[float(covariance[a, b]) for b in names] for a in names])
    except (TypeError, IndexError, KeyError):
        pass
    matrix = jnp.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(names), len(names)):
        raise ValueError(
            "covariance must support name-pair indexing covariance[a, b] "
            f"for the given `names`, or be a dense ({len(names)}, {len(names)}) "
            "matrix already ordered exactly like `names`"
        )
    return matrix


def delta_method_jacobian(
    function: Callable[[Mapping[str, object]], Array],
    values: Mapping[str, object],
    parameter_names: Sequence[str],
) -> Array:
    """Jacobian of ``function(values)`` with respect to ``parameter_names``.

    ``function`` must be a pure-JAX function of the full parameter mapping
    (for example ``PreparedAmplitudeCache.fit_fractions`` or
    ``interference_fractions``, or any other JAX-differentiable quantity
    derived from postfit parameter values), differentiated exactly via
    reverse-mode autodiff rather than a finite-difference approximation.

    ``values`` is the point at which to differentiate -- ordinarily the
    postfit values for every parameter, fixed and floating alike (e.g. from
    ``FitSession.result_values``/``CPFitSession.result_values``).
    ``parameter_names`` selects which entries of ``values`` are treated as
    variable; ordinarily the floating (non-fixed) fit parameters only, since
    a fixed parameter carries no uncertainty and so contributes nothing to
    propagated covariance regardless of ``function``'s sensitivity to it.

    One ``jax.vjp`` linearization pass computes the primal output and a
    reusable backward function; one cotangent-basis row is pulled through it
    per *output* entry, in an ordinary Python loop rather than
    ``jax.jacrev``'s default ``vmap``-batched sweep over every row at once.
    For a function like ``fit_fractions`` -- few outputs (one per component),
    but a normalization-matrix computation over a large sample that a
    reverse pass touches for every floating dynamics parameter (e.g. every
    QMI knot) -- that batching multiplies an already sizeable per-row
    intermediate by the output count *simultaneously*, which is what was
    driving this into GPU out-of-memory territory before the loop replaced
    it: each iteration's temporaries are freed before the next one starts,
    so peak memory stays at one row's cost regardless of how many outputs
    ``function`` has, at the cost of ``len(output)`` sequential backward
    passes instead of one vectorized one -- the right trade whenever, as
    here, outputs are few and each backward pass is comparatively heavy.
    """
    parameter_names = tuple(parameter_names)
    free_vector = jnp.asarray([values[name] for name in parameter_names], dtype=float)

    def vector_function(vector: Array) -> Array:
        patched = dict(values)
        patched.update(zip(parameter_names, vector))
        return jnp.atleast_1d(jnp.asarray(function(patched)))

    output, pullback = jax.vjp(vector_function, free_vector)
    rows = []
    for index in range(output.shape[0]):
        cotangent = jnp.zeros_like(output).at[index].set(1.0)
        (row,) = pullback(cotangent)
        rows.append(row)
    return jnp.stack(rows, axis=0)


def delta_method_covariance(
    function: Callable[[Mapping[str, object]], Array],
    values: Mapping[str, object],
    parameter_names: Sequence[str],
    covariance: object,
) -> Array:
    """Propagate a postfit ``covariance`` through ``function`` via the delta method.

    Returns ``J @ C @ J.T``, where ``J`` is the autodiff Jacobian of
    ``function`` with respect to ``parameter_names`` (see
    ``delta_method_jacobian``) and ``C`` is ``covariance`` restricted to
    ``parameter_names`` -- typically an ``iminuit`` postfit covariance
    matrix (e.g. ``result.covariance`` from ``Minimizer.fit``), or any
    array/object satisfying the same interface, see
    ``delta_method_jacobian``. The off-diagonal entries of the returned
    matrix are the propagated covariances between the components of
    ``function``'s output, e.g. between two fit fractions or between the
    same fit fraction computed for two different charges that share fit
    parameters.
    """
    jacobian = delta_method_jacobian(function, values, parameter_names)
    matrix = _covariance_matrix(covariance, parameter_names)
    return jacobian @ matrix @ jacobian.T


def delta_method_errors(
    function: Callable[[Mapping[str, object]], Array],
    values: Mapping[str, object],
    parameter_names: Sequence[str],
    covariance: object,
) -> Array:
    """Delta-method standard errors: ``sqrt(diag(delta_method_covariance(...)))``.

    This is a linear (Gaussian) error-propagation approximation, the same
    one implicit in Minuit's HESSE errors themselves; it can understate the
    true uncertainty for quantities that are strongly nonlinear in the fit
    parameters near the postfit point, or that sit close to a hard boundary
    (e.g. a fit fraction near zero).
    """
    variance = jnp.diag(delta_method_covariance(function, values, parameter_names, covariance))
    return jnp.sqrt(jnp.clip(variance, 0.0))


__all__ = ["delta_method_covariance", "delta_method_errors", "delta_method_jacobian"]
