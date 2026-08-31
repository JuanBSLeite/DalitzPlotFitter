"""Robust outlier handling for repeated-fit studies."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from iminuit import Minuit


@dataclass(frozen=True)
class OutlierSelection:
    """Result of a robust one-dimensional outlier selection."""

    mask: np.ndarray
    center: float
    scale: float
    threshold: float

    @property
    def n_entries(self) -> int:
        return int(self.mask.size)

    @property
    def n_kept(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def n_outliers(self) -> int:
        return self.n_entries - self.n_kept

    @property
    def outlier_fraction(self) -> float:
        if self.n_entries == 0:
            return float("nan")
        return self.n_outliers / self.n_entries


@dataclass(frozen=True)
class RobustGaussianFitResult:
    """Gaussian fit after robust outlier rejection."""

    mean: float
    sigma: float
    mean_error: float
    sigma_error: float
    valid: bool
    n_entries: int
    n_kept: int
    n_outliers: int
    outlier_fraction: float
    robust_center: float
    robust_scale: float
    threshold: float


def robust_outlier_mask(
    values,
    *,
    threshold: float = 5.0,
) -> OutlierSelection:
    """Select inliers with a median/MAD robust-sigma criterion.

    The robust scale is ``1.4826 * median(abs(x - median(x)))``. Finite points
    satisfying ``abs(x - median) <= threshold * scale`` are retained. If the
    MAD vanishes, the function falls back to the standard deviation; if that
    also vanishes, every finite entry is retained.

    ``threshold=5`` is intentionally conservative for pseudoexperiment studies:
    the goal is to remove catastrophic/minimum-swap tails, not sculpt the
    Gaussian core.
    """

    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")

    finite = np.isfinite(values)
    if not np.any(finite):
        return OutlierSelection(
            mask=np.zeros(values.shape, dtype=bool),
            center=float("nan"),
            scale=float("nan"),
            threshold=float(threshold),
        )

    finite_values = values[finite]
    center = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - center)))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(finite_values, ddof=1)) if finite_values.size > 1 else 0.0

    mask = finite.copy()
    if np.isfinite(scale) and scale > 0.0:
        mask &= np.abs(values - center) <= threshold * scale

    return OutlierSelection(
        mask=mask,
        center=center,
        scale=scale,
        threshold=float(threshold),
    )


def robust_gaussian_fit(values, *, threshold: float = 5.0) -> RobustGaussianFitResult:
    """Reject robust outliers and fit a Gaussian to the retained core."""

    values = np.asarray(values, dtype=float)
    selection = robust_outlier_mask(values, threshold=threshold)
    kept = values[selection.mask]
    kept = kept[np.isfinite(kept)]

    if kept.size < 2:
        return RobustGaussianFitResult(
            mean=float(kept[0]) if kept.size else math.nan,
            sigma=math.nan,
            mean_error=math.nan,
            sigma_error=math.nan,
            valid=False,
            n_entries=selection.n_entries,
            n_kept=int(kept.size),
            n_outliers=selection.n_outliers,
            outlier_fraction=selection.outlier_fraction,
            robust_center=selection.center,
            robust_scale=selection.scale,
            threshold=selection.threshold,
        )

    mean0 = float(np.mean(kept))
    sigma0 = float(np.std(kept, ddof=1))
    if not np.isfinite(sigma0) or sigma0 <= 0.0:
        sigma0 = max(abs(mean0), 1.0) * 1e-6

    def gaussian_nll(mean, sigma):
        if sigma <= 0.0:
            return np.inf
        z = (kept - mean) / sigma
        return float(kept.size * np.log(sigma) + 0.5 * np.sum(z * z))

    fit = Minuit(gaussian_nll, mean=mean0, sigma=sigma0)
    fit.errordef = 0.5
    fit.limits["sigma"] = (np.finfo(float).tiny, None)
    fit.migrad()
    fit.hesse()

    return RobustGaussianFitResult(
        mean=float(fit.values["mean"]),
        sigma=float(fit.values["sigma"]),
        mean_error=float(fit.errors["mean"]),
        sigma_error=float(fit.errors["sigma"]),
        valid=bool(fit.valid),
        n_entries=selection.n_entries,
        n_kept=selection.n_kept,
        n_outliers=selection.n_outliers,
        outlier_fraction=selection.outlier_fraction,
        robust_center=selection.center,
        robust_scale=selection.scale,
        threshold=selection.threshold,
    )


def genfit_distribution(result, name: str) -> np.ndarray:
    """Return the valid GenFit distribution for one parameter or the NLL."""

    if name == "nll":
        return np.asarray(result.nll[result.valid_mask], dtype=float)
    return np.asarray(result.values(name, valid_only=True), dtype=float)


def genfit_outlier_selection(
    result,
    name: str,
    *,
    threshold: float = 5.0,
) -> OutlierSelection:
    """Return the robust outlier mask for a GenFit parameter or NLL."""

    return robust_outlier_mask(
        genfit_distribution(result, name),
        threshold=threshold,
    )


def genfit_robust_gaussian_fit(
    result,
    name: str,
    *,
    threshold: float = 5.0,
) -> RobustGaussianFitResult:
    """Robust Gaussian fit for a GenFit parameter or NLL distribution."""

    return robust_gaussian_fit(
        genfit_distribution(result, name),
        threshold=threshold,
    )


def genfit_robust_summary(result, *, threshold: float = 5.0):
    """Return robust summary rows for all fitted parameters and the NLL."""

    rows = []
    for name in (*result.parameter_names, "nll"):
        values = genfit_distribution(result, name)
        selection = robust_outlier_mask(values, threshold=threshold)
        kept = values[selection.mask]
        gaussian = robust_gaussian_fit(values, threshold=threshold)
        rows.append(
            {
                "name": name,
                "entries": int(values.size),
                "kept": int(kept.size),
                "n_outliers": selection.n_outliers,
                "outlier_fraction": selection.outlier_fraction,
                "sample_mean": float(np.mean(kept)) if kept.size else math.nan,
                "sample_std": (
                    float(np.std(kept, ddof=1)) if kept.size > 1 else math.nan
                ),
                "gauss_mean": gaussian.mean,
                "gauss_mean_error": gaussian.mean_error,
                "gauss_sigma": gaussian.sigma,
                "gauss_sigma_error": gaussian.sigma_error,
                "gauss_valid": gaussian.valid,
            }
        )
    return rows
