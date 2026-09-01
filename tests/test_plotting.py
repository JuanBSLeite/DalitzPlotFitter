import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dalitzplotfitter import binned_data, plot_binned_data


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
