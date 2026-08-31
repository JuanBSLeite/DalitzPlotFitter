"""Robust outlier handling for repeated-fit studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    Parameters
    ----------
    values:
        One-dimensional values to classify.
    threshold:
        Maximum robust z-score. ``5.0`` is deliberately conservative for
        pseudoexperiment studies: it removes catastrophic/minimum-swap tails
        without sculpting an otherwise Gaussian core.
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
