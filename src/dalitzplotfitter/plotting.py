"""Small plotting helpers for common analysis diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dalitzplotfitter.kinematics import invariants_to_square_dalitz


def _values(sample, variable: str):
    if hasattr(sample, variable):
        return np.asarray(getattr(sample, variable))
    if isinstance(sample, dict) and variable in sample:
        return np.asarray(sample[variable])
    raise KeyError(f"sample does not contain {variable!r}")


def binned_data(
    values,
    *,
    bins=60,
    range: tuple[float, float] | None = None,
    weights=None,
):
    """Return bin centers, counts, statistical uncertainties and bin edges.

    Unweighted data use the usual ``sqrt(N)`` Poisson approximation. Weighted
    data use ``sqrt(sum w^2)`` in each bin.
    """

    values = np.asarray(values)
    if weights is None:
        counts, edges = np.histogram(values, bins=bins, range=range)
        errors = np.sqrt(counts.astype(float))
    else:
        weights = np.asarray(weights)
        counts, edges = np.histogram(values, bins=bins, range=range, weights=weights)
        sumw2, _ = np.histogram(values, bins=edges, weights=weights**2)
        errors = np.sqrt(sumw2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts, errors, edges


def plot_binned_data(
    values,
    *,
    bins=60,
    range: tuple[float, float] | None = None,
    weights=None,
    ax=None,
    label: str = "data",
    markersize: float = 4.5,
):
    """Plot one-dimensional data as black circular points with error bars."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    centers, counts, errors, edges = binned_data(
        values, bins=bins, range=range, weights=weights
    )
    ax.errorbar(
        centers,
        counts,
        yerr=errors,
        fmt="o",
        color="black",
        ecolor="black",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=markersize,
        linestyle="none",
        label=label,
        zorder=10,
    )
    return ax, counts, errors, edges


def plot_dalitz(
    sample,
    *,
    x: str = "s13",
    y: str = "s23",
    weights=None,
    bins: int = 70,
    ax=None,
    title: str | None = None,
    colorbar: bool = True,
):
    """Plot a standard two-dimensional Dalitz histogram in one call."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    hist = ax.hist2d(_values(sample, x), _values(sample, y), bins=bins, weights=weights)
    ax.set_xlabel(rf"${x}$ [GeV$^2$]")
    ax.set_ylabel(rf"${y}$ [GeV$^2$]")
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(hist[3], ax=ax)
    return ax


def plot_square_dalitz(
    sample,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
    pair: tuple[int, int] = (0, 1),
    weights=None,
    bins: int = 70,
    ax=None,
    title: str | None = None,
    colorbar: bool = True,
):
    """Plot a Square-Dalitz histogram from ordinary invariant coordinates."""

    data = sample.as_dict() if hasattr(sample, "as_dict") else sample
    mp, tp = invariants_to_square_dalitz(
        data["s12"], data["s13"], data["s23"],
        mother_mass=mother_mass,
        masses=masses,
        pair=pair,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    hist = ax.hist2d(
        np.asarray(mp),
        np.asarray(tp),
        bins=bins,
        weights=weights,
        range=((0, 1), (0, 1)),
    )
    ax.set_xlabel(r"$m'$")
    ax.set_ylabel(r"$\theta'$")
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(hist[3], ax=ax)
    return ax


__all__ = [
    "binned_data",
    "plot_binned_data",
    "plot_dalitz",
    "plot_square_dalitz",
]
