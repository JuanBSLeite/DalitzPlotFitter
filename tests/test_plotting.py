import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from dalitzplotfitter import (
    binned_data,
    plot_binned_data,
    plot_dalitz,
    plot_square_dalitz,
    square_dalitz_to_invariants,
)


def test_binned_data_uses_poisson_sqrt_n_for_unweighted_data():
    values = np.asarray([0.1, 0.2, 0.3, 1.1, 1.2])
    centers, counts, errors, edges = binned_data(values, bins=np.asarray([0.0, 1.0, 2.0]))

    assert np.allclose(centers, [0.5, 1.5])
    assert np.array_equal(counts, [3, 2])
    assert np.allclose(errors, np.sqrt([3.0, 2.0]))
    assert np.array_equal(edges, [0.0, 1.0, 2.0])


def test_binned_data_uses_sumw2_for_weighted_data():
    values = np.asarray([0.1, 0.2, 1.1])
    weights = np.asarray([1.0, 2.0, 3.0])
    _, counts, errors, _ = binned_data(
        values,
        bins=np.asarray([0.0, 1.0, 2.0]),
        weights=weights,
    )

    assert np.allclose(counts, [3.0, 3.0])
    assert np.allclose(errors, [np.sqrt(5.0), 3.0])


def test_plot_binned_data_defaults_to_black_circular_markers():
    fig, ax = plt.subplots()
    plot_binned_data(
        np.asarray([0.1, 0.2, 1.1]),
        bins=np.asarray([0.0, 1.0, 2.0]),
        ax=ax,
    )

    marker_lines = [line for line in ax.lines if line.get_marker() == "o"]
    assert len(marker_lines) == 1
    line = marker_lines[0]
    assert line.get_color() == "black"
    assert line.get_markerfacecolor() == "black"
    assert line.get_linestyle() in ("None", "none", "")

    plt.close(fig)


def test_plot_binned_data_can_use_log_scale_and_bin_width_ylabel():
    fig, ax = plt.subplots()
    plot_binned_data(
        np.asarray([0.1, 0.2, 0.8, 1.1, 1.2]),
        bins=np.asarray([0.0, 0.5, 1.0, 1.5]),
        ax=ax,
        unit=r"GeV$^2$",
        log_scale=True,
    )

    assert ax.get_yscale() == "log"
    assert ax.get_ylabel() == r"Candidates / 0.5 GeV$^2$"
    plt.close(fig)


def test_plot_dalitz_can_use_logarithmic_color_normalization():
    sample = {
        "s13": np.asarray([0.1, 0.2, 0.2, 0.8, 0.9]),
        "s23": np.asarray([0.2, 0.2, 0.3, 0.8, 0.9]),
    }
    fig, ax = plt.subplots()
    plot_dalitz(sample, bins=4, ax=ax, colorbar=False, log_scale=True)

    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0].norm, LogNorm)
    plt.close(fig)


def test_plot_dalitz_folded_matches_pre_folded_data():
    sample = {
        "s12": np.asarray([0.1, 0.9, 0.3, 0.7]),
        "s13": np.asarray([0.9, 0.1, 0.7, 0.3]),
    }
    fig1, ax1 = plt.subplots()
    plot_dalitz(sample, x="s12", y="s13", bins=4, ax=ax1, colorbar=False, folded=True)
    folded_counts = np.asarray(ax1.collections[0].get_array())

    pre_folded = {
        "s12": np.minimum(sample["s12"], sample["s13"]),
        "s13": np.maximum(sample["s12"], sample["s13"]),
    }
    fig2, ax2 = plt.subplots()
    plot_dalitz(pre_folded, x="s12", y="s13", bins=4, ax=ax2, colorbar=False)
    reference_counts = np.asarray(ax2.collections[0].get_array())

    assert np.array_equal(folded_counts, reference_counts)
    assert folded_counts.sum() == 4
    assert "min" in ax1.get_xlabel()
    assert "max" in ax1.get_ylabel()
    plt.close(fig1)
    plt.close(fig2)


def test_plot_square_dalitz_folded_restricts_thetaprime_to_half():
    mother_mass = 1.86966
    masses = (0.13957, 0.13957, 0.13957)
    mp = np.asarray([0.2, 0.5, 0.8, 0.3])
    tp = np.asarray([0.1, 0.9, 0.2, 0.8])
    s12, s13, s23 = square_dalitz_to_invariants(
        mp, tp, mother_mass=mother_mass, masses=masses, pair=(0, 1),
    )
    sample = {"s12": np.asarray(s12), "s13": np.asarray(s13), "s23": np.asarray(s23)}

    fig, ax = plt.subplots()
    plot_square_dalitz(
        sample, mother_mass=mother_mass, masses=masses, pair=(0, 1),
        bins=4, ax=ax, colorbar=False, folded=True,
    )
    assert ax.get_ylim()[1] <= 0.5 + 1e-9
    assert "folded" in ax.get_ylabel()
    plt.close(fig)
