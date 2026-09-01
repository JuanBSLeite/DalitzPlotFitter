"""Small plotting helpers for common analysis diagnostics."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dalitzplotfitter.kinematics import invariants_to_square_dalitz


def _values(sample, variable: str):
    if hasattr(sample, variable):
        return np.asarray(getattr(sample, variable))
    if isinstance(sample, dict) and variable in sample:
        return np.asarray(sample[variable])
    raise KeyError(f"sample does not contain {variable!r}")


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
    hist = ax.hist2d(np.asarray(mp), np.asarray(tp), bins=bins, weights=weights, range=((0, 1), (0, 1)))
    ax.set_xlabel(r"$m'$")
    ax.set_ylabel(r"$\theta'$")
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(hist[3], ax=ax)
    return ax


__all__ = ["plot_dalitz", "plot_square_dalitz"]
