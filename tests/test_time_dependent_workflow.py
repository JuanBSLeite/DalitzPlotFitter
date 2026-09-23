import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    NeutralMesonMixing,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    RealImag,
    TimeDependentFitSession,
)
from dalitzplotfitter.time_dependent_workflow import _ReflectedAmplitude


def _channel():
    return DecayChannel("D0", ("K(S)0", "pi+", "pi-"))


def _model(name="NR", value=1.0):
    x = Parameter.coefficient(f"{name}.x", value, bounds=(0.2, 2.0), owner=name)
    return DecayModel(
        _channel(),
        [NonResonant(RealImag(x, 0.0), name=name)],
        normalization_method="square-dalitz",
        normalization_resolution=12,
    )


def _data(n=6, seed=0):
    # Borrow physically-valid (s12, s13, s23) triplets from the model's own
    # normalization sample instead of hand-computing Dalitz-boundary points.
    sample = _model().normalization_sample
    rng = np.random.default_rng(seed)
    idx = rng.choice(sample.size, size=n, replace=False)
    return PhaseSpaceSample(
        s12=jnp.asarray(sample.s12)[idx],
        s13=jnp.asarray(sample.s13)[idx],
        s23=jnp.asarray(sample.s23)[idx],
        weights=jnp.ones(n),
    )


def _mixing():
    return NeutralMesonMixing(
        Parameter("mix.x", 0.01, bounds=(-0.1, 0.1)),
        Parameter("mix.y", 0.005, bounds=(-0.1, 0.1)),
        0.4103,
    )


def _fixed_mixing():
    # NonResonant is Dalitz-independent, so the auto-reflected Abar equals A
    # exactly here, and with q_over_p=1 (default) x drops out of the rate
    # entirely (see the derivation in the module docstring notes below): x is
    # a genuinely flat/unidentified direction for this toy model, not just
    # weakly constrained. Floating it (as _mixing() does) lets Migrad land
    # anywhere in its bounds, including exactly at one -- flaky across
    # platforms. Tests that only need "a session that fits", not mixing
    # convergence itself, use fixed values instead.
    return NeutralMesonMixing(0.01, 0.005, 0.4103)


def _session(**kwargs):
    data = kwargs.pop("data", _data())
    n = data.size
    times = kwargs.pop("times", jnp.linspace(0.1, 2.0, n))
    tags = kwargs.pop("tags", jnp.asarray([1, -1] * (n // 2 + 1))[:n])
    mixing = kwargs.pop("mixing", _mixing())
    model = kwargs.pop("model", _model())
    return TimeDependentFitSession(model, data, times, tags, mixing, **kwargs)


def test_reflected_amplitude_swaps_s12_s13():
    def dynamics(data, parameters=None):
        return data["s12"] - 2 * data["s13"]

    reflected = _ReflectedAmplitude(dynamics)
    data = {
        "s12": jnp.array([1.0, 2.0]),
        "s13": jnp.array([3.0, 4.0]),
        "s23": jnp.array([9.0, 9.0]),
    }
    expected = data["s13"] - 2 * data["s12"]
    np.testing.assert_allclose(reflected(data), expected)


def test_parameters_collect_model_and_mixing_deduplicated():
    session = _session()
    names = sorted(p.name for p in session.parameters)
    assert names == ["NR.x", "mix.x", "mix.y"]


def test_objective_is_finite_at_declared_values():
    session = _session()
    values = {"NR.x": 1.0, "mix.x": 0.01, "mix.y": 0.005}
    assert jnp.isfinite(session.objective(values))


def test_session_matches_hand_built_time_dependent_nll():
    from dalitzplotfitter import (
        AmplitudeComponent,
        PreparedAmplitudeCache,
        TimeDependentDalitzNLL,
    )

    session = _session()
    values = {"NR.x": 1.0, "mix.x": 0.01, "mix.y": 0.005}

    model = session.model
    a_components = model.amplitude_model.components
    abar_components = tuple(
        AmplitudeComponent(
            "bar_" + c.name, _ReflectedAmplitude(c.function), c.coefficient, False
        )
        for c in a_components
    )
    sample = model.normalization_sample
    cache = PreparedAmplitudeCache.prepare(
        a_components + abar_components,
        data=session.data.as_dict(),
        normalization_data=sample.as_dict(),
        normalization_weights=sample.weights,
        normalize_components=False,
    )
    manual_nll = TimeDependentDalitzNLL(
        cache, len(a_components), session.times, session.tags, session.mixing,
    )
    np.testing.assert_allclose(
        np.asarray(session.base_objective.densities(values)),
        np.asarray(manual_nll.densities(values)),
        rtol=1e-12,
    )


def test_abar_model_requires_matching_daughter_masses():
    mismatched = DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
    )
    with pytest.raises(ValueError):
        _session(abar_model=mismatched)


def test_explicit_abar_model_is_used_directly_not_reflected():
    abar_model = _model(name="ABAR", value=1.7)
    session = _session(abar_model=abar_model)
    abar_component = session._abar_components[0]
    # The explicit abar_model's own coefficient must be used verbatim, not a
    # reflection of `model`'s coefficient.
    expected_coefficient = abar_model.amplitude_model.components[0].coefficient
    assert abar_component.coefficient is expected_coefficient
    names = sorted(p.name for p in session.parameters)
    assert names == ["ABAR.x", "NR.x", "mix.x", "mix.y"]


def test_fit_update_model_returns_fitted_values():
    session = _session(mixing=_fixed_mixing())
    result, fitted_model = session.fit(
        {"NR.x": 0.9}, simplex=False, ncall=50, update_model=True,
    )
    assert result.valid
    fitted_x = next(p for p in fitted_model.parameters if p.name == "NR.x").value
    assert fitted_x == pytest.approx(float(result.values["NR.x"]))
    # the session's own model is untouched
    original_x = next(p for p in session.model.parameters if p.name == "NR.x")
    assert original_x.value == 1.0


def test_report_and_fit_fractions_run():
    session = _session(mixing=_fixed_mixing())
    result = session.fit({"NR.x": 0.9}, simplex=False, ncall=50)
    report = session.report(result)
    assert set(report) >= {"valid", "nll", "edm", "values", "errors", "fit_fractions"}
    assert report["fit_fractions"]["NR"] == pytest.approx(1.0, rel=1e-6)


def test_plot_time_projection_runs_and_returns_axes_with_both_tags():
    session = _session(mixing=_fixed_mixing())
    result = session.fit({"NR.x": 0.9}, simplex=False, ncall=50)
    ax = session.plot_time_projection(result)
    assert ax.get_xlabel() == "t"
    # both tags present in this session's data -> 4 series.
    assert len(ax.get_legend().get_texts()) == 4
    # each tag's data points get their own colour (not both hardcoded black),
    # so the two tags stay visually distinguishable on the same axes.
    data_containers = [c for c in ax.containers if c.get_label().startswith("data ")]
    assert len(data_containers) == 2
    assert len({c.lines[0].get_color() for c in data_containers}) == 2


def test_plot_time_projection_skips_a_tag_with_no_observed_events():
    data = _data()
    session = _session(data=data, tags=jnp.ones(data.size), mixing=_fixed_mixing())
    result = session.fit({"NR.x": 0.9}, simplex=False, ncall=50)
    ax = session.plot_time_projection(result)
    assert len(ax.get_legend().get_texts()) == 2


def test_plot_time_projection_rejects_event_wise_wrong_tag():
    data = _data()
    session = _session(
        data=data, wrong_tag=jnp.full(data.size, 0.1), mixing=_fixed_mixing(),
    )
    result = session.fit({"NR.x": 0.9}, simplex=False, ncall=50)
    with pytest.raises(ValueError, match="scalar wrong_tag"):
        session.plot_time_projection(result)


def test_plot_time_projection_uses_finite_time_range_by_default():
    data = _data()
    session = _session(data=data, time_range=(0.0, 3.0), mixing=_fixed_mixing())
    result = session.fit({"NR.x": 0.9}, simplex=False, ncall=50)
    ax = session.plot_time_projection(result)
    curve = next(line for line in ax.get_lines() if line.get_label().startswith("fit "))
    xdata = curve.get_xdata()
    assert xdata.min() == pytest.approx(0.0)
    assert xdata.max() == pytest.approx(3.0)


def test_plot_projection_runs_and_returns_two_axes():
    session = _session(mixing=_fixed_mixing())
    result = session.fit({"NR.x": 0.9}, simplex=False, ncall=50)
    axes = session.plot_projection(result, "s12", bins=10, projection_size=2_000)
    assert len(axes) == 2
    assert axes[0].get_title() == "D0"
    assert axes[1].get_title() == "D0bar"
    for ax in axes:
        assert ax.get_xlabel() == r"$s12$ [GeV$^2$]"


def test_plot_projection_with_pulls_returns_2x2_grid():
    session = _session(mixing=_fixed_mixing())
    result = session.fit({"NR.x": 0.9}, simplex=False, ncall=50)
    grid = session.plot_projection(
        result, "s13", bins=10, projection_size=2_000, show_pulls=True,
    )
    assert grid.shape == (2, 2)


def test_plot_projection_with_pulls_and_explicit_axes_raises():
    import matplotlib.pyplot as plt

    session = _session(mixing=_fixed_mixing())
    result = session.fit({"NR.x": 0.9}, simplex=False, ncall=50)
    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="axes=None"):
        session.plot_projection(result, show_pulls=True, axes=axes)


def test_plot_projection_rejects_event_wise_wrong_tag():
    data = _data()
    session = _session(
        data=data, wrong_tag=jnp.full(data.size, 0.1), mixing=_fixed_mixing(),
    )
    result = session.fit({"NR.x": 0.9}, simplex=False, ncall=50)
    with pytest.raises(ValueError, match="scalar wrong_tag"):
        session.plot_projection(result, "s12", projection_size=2_000)
