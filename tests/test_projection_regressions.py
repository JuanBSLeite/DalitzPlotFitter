from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from jaxpwa import (
    BackgroundCategory,
    CPBackgroundCategory,
    CPFitSession,
    DecayChannel,
    DecayModel,
    FitSession,
    NonResonant,
    RealImag,
    YieldAsymmetry,
)
from jaxpwa.cp_workflow import _joint_scaled_weights
from jaxpwa.workflow import _scaled_projection_weights


@pytest.fixture
def setup():
    m = DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
    )
    d = m.generate_phase_space(20, seed=2, include_momenta=False)
    yield m, d, m.normalization_sample
    plt.close("all")


def test_precomputed_backgrounds_fail_explicitly(setup):
    m, d, s = setup
    bg = BackgroundCategory("bg", jnp.ones(d.size), 1.0)
    f = FitSession(m, d, backgrounds=(bg,), signal_fraction=0.6)
    with pytest.raises(ValueError, match="BackgroundSpec"):
        f._projection_components({}, s)
    cpbg = CPBackgroundCategory("bg", jnp.ones(d.size), jnp.ones(d.size), 1.0, 1.0)
    cp = CPFitSession(m, m, d, d, backgrounds=(cpbg,), signal_fraction=0.6)
    with pytest.raises(ValueError, match="CPBackgroundSpec"):
        cp._projection_components_pair({}, s, s)


def test_cp_yields_do_not_depend_on_projection_size(setup):
    m, d, s = setup
    large = s.take(jnp.tile(jnp.arange(s.size), 4))
    cp = CPFitSession(m, m, d, d)
    p, n = cp._projection_components_pair({}, s, large)
    assert np.isclose(np.sum(p[0][2]), 20)
    assert np.isclose(np.sum(n[0][2]), 20)
    pw, mw = _joint_scaled_weights(
        s, jnp.ones(s.size), large, jnp.ones(large.size), 40.0
    )
    assert np.isclose(pw.sum(), 20)
    assert np.isclose(mw.sum(), 20)


def test_cp_empty_charge_and_common_bins(setup):
    m, d, s = setup
    empty = d.take(jnp.array([], dtype=int))
    cp = CPFitSession(m, m, d, empty)
    axes = cp.plot_projection(SimpleNamespace(values={}), projection_size=100, bins=8)
    np.testing.assert_array_equal(
        axes[0].patches[-1].get_data().edges, axes[1].patches[-1].get_data().edges
    )
    with pytest.raises(ValueError, match="provide range"):
        CPFitSession(m, m, empty, empty).plot_projection(SimpleNamespace(values={}))


def test_zero_yield_skips_density_evaluation(setup, monkeypatch):
    m, d, s = setup
    np.testing.assert_array_equal(
        _scaled_projection_weights(s, jnp.zeros(s.size), 0), np.zeros(s.size)
    )

    def fail(*args, **kwargs):
        raise AssertionError("absent signal evaluated")

    monkeypatch.setattr(FitSession, "_projection_signal_density", fail)
    f = FitSession(m, d, extended=True, signal_yield=0.0)
    assert np.sum(f._projection_components({}, s)[0][2]) == 0
    cp = CPFitSession(m, m, d, d, extended=True, signal_yield=0.0)
    p, n = cp._projection_components_pair({}, s, s)
    assert np.sum(p[0][2]) + np.sum(n[0][2]) == 0


def test_cp_background_totals_match_fitted_probabilities(setup):
    m, d, s = setup
    cp = CPFitSession(m, m, d, d, signal_fraction=0.6).with_background(
        "bg",
        lambda data: data["s12"],
        minus_shape=lambda data: jnp.ones_like(data["s12"]),
    )
    large = s.take(jnp.tile(jnp.arange(s.size), 3))
    p, n = cp._projection_components_pair({}, s, large)
    pp, pm = cp.base_objective.charge_probabilities({})
    assert np.isclose(sum(np.sum(c[2]) for c in p), 40 * float(pp))
    assert np.isclose(sum(np.sum(c[2]) for c in n), 40 * float(pm))


@pytest.mark.parametrize("standalone_yields", [False, True])
def test_cp_regional_projection_preserves_full_signal_and_background_yields(
    setup, standalone_yields,
):
    m, d, _ = setup
    signal_yield = YieldAsymmetry(80.0, 0.3) if standalone_yields else 80.0
    cp = CPFitSession(m, m, d, d, extended=True, signal_yield=signal_yield)
    cp = cp.with_background(
        "bg", lambda data: data["s23"],
        minus_shape=lambda data: jnp.ones_like(data["s23"]), yield_=20.0,
    )
    # Cut the OTHER coordinate, so an x-axis range cannot reproduce this plot.
    cutoff = float(np.median(d.s23))

    def selection(data):
        return np.asarray(data["s23"]) < cutoff

    size, seed = 500, 734
    samples = [m.generate_phase_space(size, seed=seed + q) for q in (0, 1)]
    full_components = cp._projection_components_pair({}, *samples)
    limits = (0.0, m.channel.parent_mass**2)
    grid = cp.plot_projection(
        SimpleNamespace(values={}), "s12", partner_variable="s13",
        folded=True, fold_side="high", selection=selection,
        range=limits, bins=8, projection_size=size, projection_seed=seed,
        show_pulls=True,
    )
    for column, components in enumerate(full_components):
        total = np.zeros(8)
        for patch, (_, sample, weights) in zip(
            grid[0, column].patches[:-1], components, strict=True,
        ):
            mask = selection(sample.as_dict())
            expected, edges = np.histogram(
                np.maximum(sample.s12, sample.s13)[mask], bins=8,
                range=limits, weights=np.asarray(weights)[mask],
            )
            np.testing.assert_allclose(patch.get_data().values, expected)
            total += expected
        np.testing.assert_allclose(grid[0, column].patches[-1].get_data().values, total)
        observed, _ = np.histogram(
            np.maximum(d.s12, d.s13)[selection(d.as_dict())], bins=edges,
        )
        np.testing.assert_array_equal(grid[0, column].lines[0].get_ydata(), observed)
        assert 0 < total.sum() < sum(np.sum(c[2]) for c in components)
    assert cp.plus_data is d and cp.minus_data is d


@pytest.mark.parametrize("mask", [True, [True], [1] * 20])
def test_cp_projection_rejects_invalid_region_masks(setup, mask):
    m, d, _ = setup
    cp = CPFitSession(m, m, d, d)
    with pytest.raises(ValueError, match="projection selection.*boolean vector"):
        cp.plot_projection(SimpleNamespace(values={}), selection=lambda data: mask)


def test_cp_projection_empty_region_with_explicit_range(setup):
    m, d, _ = setup
    cp = CPFitSession(m, m, d, d)
    def selection(data):
        return np.zeros(len(data["s13"]), dtype=bool)
    with pytest.raises(ValueError, match="provide range"):
        cp.plot_projection(SimpleNamespace(values={}), selection=selection)
    grid = cp.plot_projection(
        SimpleNamespace(values={}), selection=selection, range=(0.0, 4.0),
        projection_size=100, bins=8, show_pulls=True,
    )
    for ax in grid[0]:
        np.testing.assert_array_equal(ax.patches[-1].get_data().values, np.zeros(8))
