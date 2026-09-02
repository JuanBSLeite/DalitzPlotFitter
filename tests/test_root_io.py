import numpy as np
import jax.numpy as jnp
import uproot

from dalitzplotfitter import (
    PhaseSpaceSample,
    histogram_background_from_root,
    histogram_efficiency_from_root,
    read_phase_space_sample,
    read_root_histogram2d,
    read_root_tree,
    write_cp_phase_space_sample,
    write_phase_space_sample,
)


def test_read_root_tree_and_phase_space(tmp_path):
    path = tmp_path / "events.root"
    with uproot.recreate(path) as root_file:
        root_file["DecayTree"] = {
            "S12": np.asarray([1.0, 1.1, 1.2]),
            "S13": np.asarray([2.0, 2.1, 2.2]),
            "S23": np.asarray([3.0, 3.1, 3.2]),
            "W": np.asarray([1.0, 0.5, 2.0]),
        }

    arrays = read_root_tree(path, "DecayTree", {"s13": "S13", "weight": "W"})
    assert jnp.allclose(arrays["s13"], jnp.asarray([2.0, 2.1, 2.2]))
    assert jnp.allclose(arrays["weight"], jnp.asarray([1.0, 0.5, 2.0]))

    sample = read_phase_space_sample(
        path,
        "DecayTree",
        s12="S12",
        s13="S13",
        s23="S23",
        weight="W",
    )
    assert sample.size == 3
    assert jnp.allclose(sample.s12, jnp.asarray([1.0, 1.1, 1.2]))
    assert jnp.allclose(sample.weights, jnp.asarray([1.0, 0.5, 2.0]))


def test_write_phase_space_sample(tmp_path):
    sample = PhaseSpaceSample(
        s12=jnp.asarray([1.0, 1.1]),
        s13=jnp.asarray([2.0, 2.1]),
        s23=jnp.asarray([3.0, 3.1]),
        weights=jnp.ones(2),
    )
    path = tmp_path / "toy.root"
    write_phase_space_sample(path, sample)
    with uproot.open(path) as root_file:
        tree = root_file["DecayTree"]
        assert tree.classname == "TTree"
        assert tree.num_entries == 2
        arrays = tree.arrays(["s12", "s13", "s23", "weight"], library="np")
        assert np.allclose(arrays["s13"], [2.0, 2.1])
        assert np.allclose(arrays["weight"], 1.0)


def test_write_cp_sample_uses_one_tree_with_charge_branch(tmp_path):
    plus = PhaseSpaceSample(
        s12=jnp.asarray([1.0, 1.1, 1.2]),
        s13=jnp.asarray([2.0, 2.1, 2.2]),
        s23=jnp.asarray([3.0, 3.1, 3.2]),
        weights=jnp.ones(3),
    )
    minus = PhaseSpaceSample(
        s12=jnp.asarray([1.3, 1.4]),
        s13=jnp.asarray([2.3, 2.4]),
        s23=jnp.asarray([3.3, 3.4]),
        weights=jnp.ones(2),
    )
    path = tmp_path / "cp_toy.root"
    write_cp_phase_space_sample(path, plus, minus)

    with uproot.open(path) as root_file:
        assert "DecayTree" in root_file
        tree = root_file["DecayTree"]
        assert tree.classname == "TTree"
        assert tree.num_entries == 5
        arrays = tree.arrays(["charge", "s13"], library="np")
        assert np.array_equal(arrays["charge"], np.asarray([1, 1, 1, -1, -1], dtype=np.int8))
        assert np.count_nonzero(arrays["charge"] == 1) == plus.size
        assert np.count_nonzero(arrays["charge"] == -1) == minus.size


def test_read_root_histogram_models(tmp_path):
    path = tmp_path / "maps.root"
    x_edges = np.asarray([0.0, 1.0, 2.0])
    y_edges = np.asarray([0.0, 2.0, 4.0])
    efficiency_values = np.asarray([[0.5, 0.7], [0.8, 0.9]])
    background_values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    with uproot.recreate(path) as root_file:
        root_file["efficiency"] = (efficiency_values, x_edges, y_edges)
        root_file["background"] = (background_values, x_edges, y_edges)

    values, xe, ye = read_root_histogram2d(path, "efficiency")
    assert jnp.allclose(values, efficiency_values)
    assert jnp.allclose(xe, x_edges)
    assert jnp.allclose(ye, y_edges)

    efficiency = histogram_efficiency_from_root(
        path, "efficiency", x_variable="s13", y_variable="s23"
    )
    background = histogram_background_from_root(
        path, "background", x_variable="s13", y_variable="s23"
    )
    data = {"s13": jnp.asarray([0.5, 1.5]), "s23": jnp.asarray([1.0, 3.0])}
    assert jnp.allclose(efficiency(data), jnp.asarray([0.5, 0.9]))
    assert jnp.allclose(background(data), jnp.asarray([1.0, 4.0]))
