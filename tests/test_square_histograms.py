import jax.numpy as jnp
import numpy as np
import uproot

from dalitzplotfitter import (
    SquareDalitzHistogramBackground,
    SquareDalitzHistogramEfficiency,
    square_dalitz_background_from_root,
    square_dalitz_efficiency_from_root,
    square_dalitz_to_invariants,
)


def _data(mp, tp):
    s12, s13, s23 = square_dalitz_to_invariants(
        jnp.asarray(mp), jnp.asarray(tp),
        mother_mass=5.27934,
        masses=(0.493677, 0.13957039, 0.13957039),
        pair=(0, 2),
    )
    return {"s12": s12, "s13": s13, "s23": s23}


def test_square_histogram_evaluates_expected_bins():
    edges = jnp.asarray([0.0, 0.5, 1.0])
    values = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    eff = SquareDalitzHistogramEfficiency(
        edges, edges, values, 5.27934, (0.493677, 0.13957039, 0.13957039), (0, 2)
    )
    got = eff(_data([0.25, 0.75], [0.25, 0.75]))
    assert jnp.allclose(got, jnp.asarray([1.0, 4.0]))


def test_square_background_uses_same_coordinate_map():
    edges = jnp.asarray([0.0, 0.5, 1.0])
    values = jnp.asarray([[0.5, 1.5], [2.5, 3.5]])
    bkg = SquareDalitzHistogramBackground(
        edges, edges, values, 5.27934, (0.493677, 0.13957039, 0.13957039), (0, 2)
    )
    got = bkg(_data([0.25, 0.75], [0.75, 0.25]))
    assert jnp.allclose(got, jnp.asarray([1.5, 2.5]))


def test_square_histograms_load_from_root(tmp_path):
    path = tmp_path / "maps.root"
    edges = np.asarray([0.0, 0.5, 1.0])
    eff_values = np.asarray([[0.7, 0.8], [0.9, 1.0]])
    bkg_values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    with uproot.recreate(path) as root_file:
        root_file["eff_sdp"] = (eff_values, edges, edges)
        root_file["bkg_sdp"] = (bkg_values, edges, edges)

    kwargs = dict(
        mother_mass=5.27934,
        masses=(0.493677, 0.13957039, 0.13957039),
        pair=(0, 2),
    )
    eff = square_dalitz_efficiency_from_root(path, "eff_sdp", **kwargs)
    bkg = square_dalitz_background_from_root(path, "bkg_sdp", **kwargs)
    data = _data([0.25, 0.75], [0.25, 0.75])
    assert jnp.allclose(eff(data), jnp.asarray([0.7, 1.0]))
    assert jnp.allclose(bkg(data), jnp.asarray([1.0, 4.0]))
