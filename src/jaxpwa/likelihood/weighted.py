"""Weighted unbinned likelihood objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
import numpy as np
from jax import Array

Parameters = Mapping[str, Array | float]
LogPDF = Callable[[dict[str, Array], Parameters], Array]


@dataclass(frozen=True)
class WeightedUnbinnedNLL:
    """Weighted objective ``-sum_i w_i log p(x_i)``.

    Negative finite weights are permitted, which is useful for diagnostic or
    externally weighted objectives. The low-level objective only defines the
    weighted NLL. High-level :class:`jaxpwa.workflow.FitSession` can apply the
    explicit squared-weight Hessian covariance correction with
    ``fit(weights=..., covariance="sweight")``.
    """

    logpdf: LogPDF
    data: dict[str, Array]
    weights: Array

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("weighted likelihood data must be non-empty")
        size = int(jnp.asarray(next(iter(self.data.values()))).shape[0])
        weights = jnp.asarray(self.weights)
        if weights.shape != (size,):
            raise ValueError(f"weights must have shape ({size},), got {weights.shape}")
        if not bool(jnp.all(jnp.isfinite(weights))):
            raise ValueError("weights must be finite")
        object.__setattr__(self, "weights", weights)

    def __call__(self, parameters: Parameters) -> Array:
        values = jnp.asarray(self.logpdf(self.data, parameters))
        if values.shape != self.weights.shape:
            raise ValueError(
                f"logpdf must return shape {self.weights.shape}, got {values.shape}"
            )
        return -jnp.sum(self.weights * values)


def sweight_covariance_from_hessians(
    weighted_hessian,
    squared_weight_hessian,
) -> np.ndarray:
    r"""Return the squared-weight Hessian covariance correction.

    For a weighted log-likelihood, define

    .. math::

       H_w = -\\sum_i w_i\\,\\partial^2 \\log p_i, \\qquad
       H_{w^2} = -\\sum_i w_i^2\\,\\partial^2 \\log p_i.

    This function returns the commonly used SumW2/RooFit-style correction

    .. math::

       C = H_w^{-1} H_{w^2} H_w^{-1}.

    The two Hessians must be evaluated at the same fitted parameter point.
    This correction is widely used for sWeight fits, but is not the most
    general asymptotically-correct covariance when the score/Hessian identity
    fails, and it does not propagate uncertainty from the determination of the
    event weights themselves. See C. Langenbruch, Eur. Phys. J. C 82 (2022)
    393, arXiv:1911.01303, Eqs. (20--21).
    """

    hessian = np.asarray(weighted_hessian, dtype=float)
    squared = np.asarray(squared_weight_hessian, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("weighted_hessian must be a square matrix")
    if squared.shape != hessian.shape:
        raise ValueError(
            "squared_weight_hessian must have the same shape as weighted_hessian"
        )
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(squared)):
        raise ValueError("sWeight covariance Hessians must be finite")

    # Numerical Hessians/HVP assembly can leave tiny antisymmetric roundoff.
    hessian = 0.5 * (hessian + hessian.T)
    squared = 0.5 * (squared + squared.T)
    try:
        left = np.linalg.solve(hessian, squared)
        covariance = np.linalg.solve(hessian, left.T).T
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "weighted Hessian is singular; cannot compute sWeight covariance"
        ) from exc
    return 0.5 * (covariance + covariance.T)
