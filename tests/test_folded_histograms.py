import jax.numpy as jnp
import pytest

from dalitzplotfitter import (
    SquareDalitzHistogramBackground,
    SquareDalitzHistogramEfficiency,
    enable_x64,
    fold_thetaprime,
    square_dalitz_to_invariants,
)
from dalitzplotfitter.background import HistogramBackground
from dalitzplotfitter.efficiency import HistogramEfficiency

enable_x64()


MOTHER_MASS = 5.27934
MASSES = (0.493677, 0.13957039, 0.13957039)


def _square_dalitz_data(mp, tp):
    s12, s13, s23 = square_dalitz_to_invariants(
        jnp.asarray(mp), jnp.asarray(tp),
        mother_mass=MOTHER_MASS, masses=MASSES, pair=(0, 2),
    )
    return {"s12": s12, "s13": s13, "s23": s23}


def test_fold_thetaprime_maps_both_halves_onto_the_same_value():
    tp = jnp.asarray([0.1, 0.3, 0.49, 0.5, 0.6, 0.9])
    folded = fold_thetaprime(tp)
    assert jnp.allclose(folded, jnp.asarray([0.1, 0.3, 0.49, 0.5, 0.4, 0.1]))
    assert bool(jnp.all(folded <= 0.5))


def test_square_dalitz_histogram_efficiency_folded_matches_across_the_fold():
    mprime_edges = jnp.asarray([0.0, 0.5, 1.0])
    thetaprime_edges = jnp.asarray([0.0, 0.25, 0.5])
    values = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    eff = SquareDalitzHistogramEfficiency(
        mprime_edges=mprime_edges, thetaprime_edges=thetaprime_edges, values=values,
        mother_mass=MOTHER_MASS, masses=MASSES, pair=(0, 2), folded=True,
    )
    low = eff(_square_dalitz_data([0.6], [0.1]))
    high = eff(_square_dalitz_data([0.6], [0.9]))
    assert jnp.allclose(low, high)
    assert float(low[0]) > 0.0


def test_square_dalitz_histogram_background_folded_matches_unfolded_at_low_half():
    edges_half = jnp.asarray([0.0, 0.25, 0.5])
    edges_full = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    values_half = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    values_full = jnp.asarray(
        [[1.0, 2.0, 2.0, 1.0], [3.0, 4.0, 4.0, 3.0]]
    )
    folded_bkg = SquareDalitzHistogramBackground(
        mprime_edges=edges_half, thetaprime_edges=edges_half, values=values_half,
        mother_mass=MOTHER_MASS, masses=MASSES, pair=(0, 2), folded=True,
    )
    unfolded_bkg = SquareDalitzHistogramBackground(
        mprime_edges=edges_half, thetaprime_edges=edges_full, values=values_full,
        mother_mass=MOTHER_MASS, masses=MASSES, pair=(0, 2), folded=False,
    )
    data = _square_dalitz_data([0.1, 0.1, 0.1, 0.1], [0.1, 0.4, 0.6, 0.9])
    assert jnp.allclose(folded_bkg(data), unfolded_bkg(data))


def test_square_dalitz_histogram_folded_rejects_edges_beyond_half():
    edges = jnp.asarray([0.0, 0.5, 1.0])
    values = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match=r"\[0, 0\.5\]"):
        SquareDalitzHistogramEfficiency(
            mprime_edges=edges, thetaprime_edges=edges, values=values,
            mother_mass=MOTHER_MASS, masses=MASSES, pair=(0, 2), folded=True,
        )


def test_histogram_efficiency_folded_gives_identical_values_across_the_fold():
    edges = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    values = jnp.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    eff = HistogramEfficiency(
        x_edges=edges, y_edges=edges, values=values,
        x_variable="s12", y_variable="s13", folded=True,
    )
    low_high = eff({"s12": jnp.asarray([0.5]), "s13": jnp.asarray([2.5])})
    high_low = eff({"s12": jnp.asarray([2.5]), "s13": jnp.asarray([0.5])})
    assert jnp.allclose(low_high, high_low)


def test_histogram_efficiency_folded_rejects_mismatched_edges():
    x_edges = jnp.asarray([0.0, 1.0, 2.0])
    y_edges = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    values = jnp.zeros((2, 3))
    with pytest.raises(ValueError, match="identical"):
        HistogramEfficiency(
            x_edges=x_edges, y_edges=y_edges, values=values, folded=True,
        )


def test_histogram_background_folded_gives_identical_values_across_the_fold():
    edges = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    values = jnp.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    bkg = HistogramBackground(
        x_edges=edges, y_edges=edges, values=values,
        x_variable="s12", y_variable="s13", folded=True,
    )
    low_high = bkg({"s12": jnp.asarray([0.5]), "s13": jnp.asarray([2.5])})
    high_low = bkg({"s12": jnp.asarray([2.5]), "s13": jnp.asarray([0.5])})
    assert jnp.allclose(low_high, high_low)


def test_histogram_background_folded_rejects_mismatched_edges():
    x_edges = jnp.asarray([0.0, 1.0, 2.0])
    y_edges = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    values = jnp.zeros((2, 3))
    with pytest.raises(ValueError, match="identical"):
        HistogramBackground(
            x_edges=x_edges, y_edges=y_edges, values=values, folded=True,
        )
