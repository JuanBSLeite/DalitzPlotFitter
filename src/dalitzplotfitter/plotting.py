"""Small plotting helpers for common analysis diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from dalitzplotfitter.kinematics import fold_thetaprime, invariants_to_square_dalitz


def _values(sample, variable: str):
    if hasattr(sample, variable):
        return np.asarray(getattr(sample, variable))
    if isinstance(sample, dict) and variable in sample:
        return np.asarray(sample[variable])
    raise KeyError(f"sample does not contain {variable!r}")


def _bin_width_label(edges, unit: str) -> str:
    widths = np.diff(np.asarray(edges, dtype=float))
    if widths.size == 0:
        return "Candidates / bin"
    if not np.allclose(widths, widths[0], rtol=1e-10, atol=1e-12):
        return "Candidates / bin"
    width = f"{float(widths[0]):.3g}"
    return f"Candidates / {width} {unit}" if unit else f"Candidates / {width}"


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
    unit: str | None = None,
    log_scale: bool = False,
):
    """Plot one-dimensional data as black circular points with error bars.

    When ``unit`` is supplied, the y-axis label includes the uniform bin width,
    for example ``Candidates / 0.25 GeV^2``. ``log_scale=True`` enables a
    logarithmic y axis.
    """

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
    if unit is not None:
        ax.set_ylabel(_bin_width_label(edges, unit))
    if log_scale:
        ax.set_yscale("log")
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
    log_scale: bool = False,
    folded: bool = False,
):
    """Plot a standard two-dimensional Dalitz histogram in one call.

    Set ``log_scale=True`` to display the bin contents with logarithmic color
    normalization. Set ``folded=True`` when ``x`` and ``y`` are two
    exchange-symmetric invariants (two identical daughters sharing the third,
    bachelor particle) to plot only the physically distinct half
    ``x <= y``, folding each point via ``min``/``max`` first.
    """

    x_values = _values(sample, x)
    y_values = _values(sample, y)
    if folded:
        x_values, y_values = (
            np.minimum(x_values, y_values),
            np.maximum(x_values, y_values),
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    hist = ax.hist2d(
        x_values,
        y_values,
        bins=bins,
        weights=weights,
        norm=LogNorm() if log_scale else None,
    )
    if folded:
        ax.set_xlabel(rf"min(${x}$, ${y}$) [GeV$^2$]")
        ax.set_ylabel(rf"max(${x}$, ${y}$) [GeV$^2$]")
    else:
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
    log_scale: bool = False,
    folded: bool = False,
):
    """Plot a Square-Dalitz histogram from ordinary invariant coordinates.

    Set ``log_scale=True`` to display the bin contents with logarithmic color
    normalization. Set ``folded=True`` when ``pair`` is built from two
    identical daughters to fold ``theta'`` onto ``[0, 0.5]``
    (:func:`~dalitzplotfitter.kinematics.fold_thetaprime`), plotting only the
    physically distinct half.
    """

    data = sample.as_dict() if hasattr(sample, "as_dict") else sample
    mp, tp = invariants_to_square_dalitz(
        data["s12"], data["s13"], data["s23"],
        mother_mass=mother_mass,
        masses=masses,
        pair=pair,
    )
    if folded:
        tp = fold_thetaprime(tp)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    hist = ax.hist2d(
        np.asarray(mp),
        np.asarray(tp),
        bins=bins,
        weights=weights,
        range=((0, 1), (0, 0.5) if folded else (0, 1)),
        norm=LogNorm() if log_scale else None,
    )
    ax.set_xlabel(r"$m'$")
    ax.set_ylabel(r"$\theta'$ (folded)" if folded else r"$\theta'$")
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
