"""End-to-end check that folded DP/SqDP efficiency/background compose with
FitSession and CPFitSession, for a channel with two identical final-state
particles (Ds+ -> pi- pi+ pi+, symmetric under exchanging the two pi+)."""

import types

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    BackgroundSpec,
    CPBackgroundSpec,
    CPFitSession,
    CPRealImag,
    DecayChannel,
    DecayModel,
    FitSession,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    RealImag,
    SquareDalitzHistogramEfficiency,
    enable_x64,
)
from dalitzplotfitter.background import HistogramBackground
from dalitzplotfitter.efficiency import HistogramEfficiency

enable_x64()

MOTHER_MASS = 1.96834
MASSES = (0.13957039, 0.13957039, 0.13957039)


def _folded_efficiency():
    edges = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
    values = jnp.asarray([
        [0.9, 0.8, 0.7, 0.6],
        [0.8, 0.9, 0.8, 0.7],
        [0.7, 0.8, 0.9, 0.8],
        [0.6, 0.7, 0.8, 0.9],
    ])
    return HistogramEfficiency(
        x_edges=edges, y_edges=edges, values=values,
        x_variable="s12", y_variable="s13", folded=True,
    )


def _folded_background():
    edges = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
    values = jnp.abs(jnp.asarray([
        [1.0, 1.2, 0.9, 1.1],
        [1.2, 1.0, 1.1, 0.9],
        [0.9, 1.1, 1.0, 1.2],
        [1.1, 0.9, 1.2, 1.0],
    ]))
    return HistogramBackground(
        x_edges=edges, y_edges=edges, values=values,
        x_variable="s12", y_variable="s13", folded=True,
    )


def _folded_square_dalitz_efficiency():
    # pair=(1, 2): the two identical pi+ in "D+" -> (pi-, pi+, pi+) (indices
    # 0, 1, 2), so m' is already symmetric and only theta' needs folding.
    mp_edges = jnp.linspace(0.0, 1.0, 5)
    tp_edges = jnp.asarray([0.0, 0.25, 0.5])
    values = jnp.asarray([[0.9, 0.8], [0.8, 0.9], [0.7, 0.8], [0.8, 0.7]])
    return SquareDalitzHistogramEfficiency(
        mprime_edges=mp_edges, thetaprime_edges=tp_edges, values=values,
        mother_mass=MOTHER_MASS, masses=MASSES, pair=(1, 2), folded=True,
    )


def _model():
    x = Parameter.coefficient("NR.x", 1.0, bounds=(0.2, 2.0), owner="NR")
    return DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(RealImag(x, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=16,
    )


def _data():
    return PhaseSpaceSample(
        s12=jnp.asarray([0.3, 0.6, 1.1, 0.4]),
        s13=jnp.asarray([1.1, 0.7, 0.4, 1.3]),
        s23=jnp.asarray([2.2, 2.3, 2.1, 1.9]),
        weights=jnp.ones(4),
    )


def test_fit_session_runs_with_folded_histogram_efficiency_and_background():
    session = FitSession(
        _model(),
        _data(),
        efficiency=_folded_efficiency(),
        backgrounds=[
            BackgroundSpec("comb", _folded_background(), fraction=None, yield_=None),
        ],
        signal_fraction=Parameter("signal_fraction", 0.8, bounds=(0.0, 1.0)),
    )
    value = session.objective({"NR.x": 1.0, "signal_fraction": 0.8})
    assert jnp.isfinite(value)

    result = session.fit(
        {"NR.x": 0.9, "signal_fraction": 0.7}, simplex=False, ncall=100
    )
    assert result.valid


def test_fit_session_runs_with_folded_square_dalitz_efficiency():
    session = FitSession(
        _model(), _data(), efficiency=_folded_square_dalitz_efficiency()
    )
    result = session.fit({"NR.x": 0.9}, simplex=False, ncall=100)
    assert result.valid


def _cp_models():
    x = Parameter.coefficient("NR.x", 1.0, bounds=(0.2, 2.0), owner="NR")
    dx = Parameter.coefficient("NR.dx", 0.05, bounds=(-0.5, 0.5), owner="NR")
    cp = CPRealImag(x, 0.0, dx, 0.0)
    plus = DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(cp.for_charge(+1))],
        normalization_method="square-dalitz",
        normalization_resolution=16,
    )
    minus = DecayModel(
        DecayChannel("D-", ("pi+", "pi-", "pi-")),
        [NonResonant(cp.for_charge(-1))],
        normalization_method="square-dalitz",
        normalization_resolution=16,
    )
    return plus, minus


def _cp_data(offset=0.0):
    return PhaseSpaceSample(
        s12=jnp.asarray([0.3 + offset, 0.6 + offset]),
        s13=jnp.asarray([1.1 - offset, 0.7 - offset]),
        s23=jnp.asarray([2.2, 2.3]),
        weights=jnp.ones(2),
    )


def test_cp_fit_session_runs_with_shared_folded_efficiency_and_background():
    plus, minus = _cp_models()
    efficiency = _folded_efficiency()
    background = CPBackgroundSpec("comb", _folded_background())
    session = (
        CPFitSession(
            plus,
            minus,
            _cp_data(),
            _cp_data(0.05),
            backgrounds=(background,),
            signal_fraction=Parameter("signal_fraction", 0.8, bounds=(0.0, 1.0)),
        )
        .with_efficiency(efficiency)
    )
    value = session.objective({"NR.x": 1.0, "NR.dx": 0.05, "signal_fraction": 0.8})
    assert jnp.isfinite(value)

    result = session.fit(
        {"NR.x": 0.9, "NR.dx": 0.02, "signal_fraction": 0.7}, simplex=False, ncall=100
    )
    assert result.valid


_FAKE_RESULT = types.SimpleNamespace(values={}, errors={}, valid=True)


def _data_line_counts(ax):
    return np.asarray(ax.containers[0].lines[0].get_ydata())


def _parameter_free_model(channel_final_state):
    # No Parameter objects in the coefficient, so `session.parameters` is
    # empty and `FitSession.result_values(_FAKE_RESULT)` needs no fit values.
    return DecayModel(
        DecayChannel("D+", channel_final_state),
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
    )


def test_fit_session_plot_projection_folded_requires_partner_variable():
    model = _parameter_free_model(("pi-", "pi+", "pi+"))
    data = model.generate_phase_space(20, seed=2)
    session = FitSession(model, data)
    with pytest.raises(ValueError, match="partner_variable"):
        session.plot_projection(_FAKE_RESULT, "s12", folded=True)


def test_fit_session_plot_projection_folded_rejects_bad_fold_side():
    model = _parameter_free_model(("pi-", "pi+", "pi+"))
    data = model.generate_phase_space(20, seed=2)
    session = FitSession(model, data)
    with pytest.raises(ValueError, match="fold_side"):
        session.plot_projection(
            _FAKE_RESULT, "s12", folded=True, partner_variable="s13",
            fold_side="middle",
        )


def test_fit_session_plot_projection_folded_projects_min_max_per_event():
    model = _parameter_free_model(("pi-", "pi+", "pi+"))
    data = model.generate_phase_space(20, seed=2)
    session = FitSession(model, data)
    ax_unfolded = session.plot_projection(_FAKE_RESULT, "s12", bins=8)
    ax_low = session.plot_projection(
        _FAKE_RESULT, "s12", bins=8, folded=True, partner_variable="s13",
        fold_side="low",
    )
    ax_high = session.plot_projection(
        _FAKE_RESULT, "s12", bins=8, folded=True, partner_variable="s13",
        fold_side="high",
    )
    # min/max keep one entry per event, so neither histogram doubles counts.
    assert np.sum(_data_line_counts(ax_unfolded)) == data.size
    assert np.sum(_data_line_counts(ax_low)) == data.size
    assert np.sum(_data_line_counts(ax_high)) == data.size

    s12 = np.asarray(data.s12)
    s13 = np.asarray(data.s13)
    low_edges = np.linspace(
        float(np.minimum(s12, s13).min()), float(np.minimum(s12, s13).max()), 9
    )
    expected_low, _ = np.histogram(np.minimum(s12, s13), bins=low_edges)
    assert np.array_equal(_data_line_counts(ax_low), expected_low)

    high_edges = np.linspace(
        float(np.maximum(s12, s13).min()), float(np.maximum(s12, s13).max()), 9
    )
    expected_high, _ = np.histogram(np.maximum(s12, s13), bins=high_edges)
    assert np.array_equal(_data_line_counts(ax_high), expected_high)

    assert "s_{\\mathrm{low}}" in ax_low.get_xlabel()
    assert "s_{\\mathrm{high}}" in ax_high.get_xlabel()


def test_cp_fit_session_plot_projection_folded_requires_partner_variable():
    plus = _parameter_free_model(("pi-", "pi+", "pi+"))
    minus = _parameter_free_model(("pi+", "pi-", "pi-"))
    plus_data = plus.generate_phase_space(20, seed=2)
    minus_data = minus.generate_phase_space(20, seed=3)
    session = CPFitSession(plus, minus, plus_data, minus_data)
    with pytest.raises(ValueError, match="partner_variable"):
        session.plot_projection(_FAKE_RESULT, "s12", folded=True)


def test_cp_fit_session_plot_projection_folded_projects_min_per_event_per_charge():
    plus = _parameter_free_model(("pi-", "pi+", "pi+"))
    minus = _parameter_free_model(("pi+", "pi-", "pi-"))
    plus_data = plus.generate_phase_space(20, seed=2)
    minus_data = minus.generate_phase_space(24, seed=3)
    session = CPFitSession(plus, minus, plus_data, minus_data)
    axes_folded = session.plot_projection(
        _FAKE_RESULT, "s12", bins=8, folded=True, partner_variable="s13",
        fold_side="low",
    )
    # min() keeps one entry per event -- neither charge's panel doubles counts.
    assert np.sum(_data_line_counts(axes_folded[0])) == plus_data.size
    assert np.sum(_data_line_counts(axes_folded[1])) == minus_data.size
    for ax in axes_folded:
        assert "s_{\\mathrm{low}}" in ax.get_xlabel()
