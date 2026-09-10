from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dalitzplotfitter import (
    BackgroundCategory,
    CPBackgroundCategory,
    CPFitSession,
    DecayChannel,
    DecayModel,
    FitSession,
    NonResonant,
    RealImag,
)
from dalitzplotfitter.cp_workflow import _joint_scaled_weights
from dalitzplotfitter.workflow import _scaled_projection_weights


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
