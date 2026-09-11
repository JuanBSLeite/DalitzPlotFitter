from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    BackgroundSpec,
    BinnedChi2Result,
    DecayChannel,
    DecayModel,
    FitSession,
    GaussianConstraint,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    PointToPointResult,
    RealImag,
    generate_signal_toy,
)


def _model():
    x = Parameter.coefficient("NR.x", 1.0, bounds=(0.2, 2.0), owner="NR")
    return DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(RealImag(x, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
    )


def _data():
    return PhaseSpaceSample(
        s12=jnp.asarray([0.2, 0.3]),
        s13=jnp.asarray([0.4, 0.5]),
        s23=jnp.asarray([2.0, 1.8]),
        weights=jnp.ones(2),
    )


def test_fit_session_signal_only_collects_model_parameters():
    session = FitSession(_model(), _data())
    assert [p.name for p in session.parameters] == ["NR.x"]
    value = session.objective({"NR.x": 1.0})
    assert jnp.isfinite(value)


def test_fit_session_cold_fit_materializes_cache_before_jit():
    session = FitSession(_model(), _data())

    # This deliberately calls fit() without touching objective or signal_cache
    # first. The prepared cache must be built outside the Minimizer JIT trace.
    result = session.fit(
        {"NR.x": 0.9},
        simplex=False,
        ncall=100,
    )

    assert result.valid
    assert "signal_cache" in session.__dict__
    assert "acceptance_data" in session.__dict__


def test_fit_session_cached_signal_matches_generic_pdf():
    session = FitSession(_model(), _data())
    values = {"NR.x": 1.3}
    cached = session._cached_signal_density(values)
    generic = session.signal_pdf(session.data.as_dict(), values)
    assert jnp.allclose(cached, generic, rtol=1e-12, atol=1e-12)


def test_fit_session_reuses_prepared_signal_cache():
    session = FitSession(_model(), _data())
    first = session.signal_cache
    second = session.signal_cache
    assert first is second
    assert first.data_components.shape[0] == session.data.size


def test_fit_session_reuses_projection_sample_for_same_size_and_seed():
    session = FitSession(_model(), _data())
    first = session._get_projection_sample(128, 1234)
    second = session._get_projection_sample(128, 1234)
    different = session._get_projection_sample(128, 1235)

    assert first is second
    assert different is not first


def test_fit_session_projection_prepared_density_matches_generic_pdf():
    session = FitSession(_model(), _data())
    sample = session._get_projection_sample(128, 5678)
    values = {"NR.x": 1.3}

    prepared = session._projection_signal_density(sample, values)
    generic = session.signal_pdf(sample.as_dict(), values)

    assert jnp.allclose(prepared, generic, rtol=1e-12, atol=1e-12)
    assert len(session._projection_prepared) == 1

    again = session._projection_signal_density(sample, values)
    assert jnp.allclose(again, prepared, rtol=1e-12, atol=1e-12)
    assert len(session._projection_prepared) == 1


def test_fit_session_automatically_normalizes_background_shape():
    model = _model()
    data = _data()
    fraction = Parameter("signal_fraction", 0.7, bounds=(0.0, 1.0))
    session = FitSession(
        model,
        data,
        backgrounds=(BackgroundSpec("comb", lambda d: jnp.ones_like(d["s12"])),),
        signal_fraction=fraction,
    )
    category = session.background_categories[0]
    expected = jnp.mean(model.normalization_sample.weights)
    assert jnp.allclose(category.normalization, expected)
    assert any(p.name == "signal_fraction" for p in session.parameters)
    assert jnp.isfinite(session.objective({"NR.x": 1.0, "signal_fraction": 0.7}))


def test_fit_session_adds_external_constraints():
    model = _model()
    constraint = GaussianConstraint(model.parameters[0], mean=1.0, sigma=0.2)
    base = FitSession(model, _data())
    constrained = base.with_constraint(constraint)
    assert jnp.allclose(
        constrained.objective({"NR.x": 1.2}) - base.objective({"NR.x": 1.2}),
        0.5,
    )


def test_fit_session_projection_weights_reproduce_expected_events():
    session = FitSession(_model(), _data())
    components = session._projection_components({"NR.x": 1.0})
    assert len(components) == 1
    assert jnp.allclose(
        jnp.sum(jnp.asarray(components[0][2])),
        session.data.size,
        rtol=1e-6,
    )


def test_sessions_reuse_normalization_without_acceptance(monkeypatch):
    model = _model()
    first = FitSession(model, model.generate_phase_space(16, seed=401))
    _ = first.signal_cache
    scales, matrix = model._fixed_normalization_templates[True]

    def unexpected_prepare(**kwargs):
        raise AssertionError('Normalization must not be prepared again')

    # A reused template only needs the data kernel, never the full norm kernel.
    monkeypatch.setattr(DecayModel, '_compact_prepare_kernel',
                        lambda self, **kwargs: unexpected_prepare(**kwargs))
    second = FitSession(model, model.generate_phase_space(16, seed=402))
    assert jnp.allclose(second.signal_cache.normalization_matrix_fixed, matrix)
    assert jnp.allclose(second.signal_cache.component_scales, scales)
    assert 'acceptance_normalization' not in second.__dict__
    assert jnp.allclose(second._cached_signal_density({'NR.x': 1.0}),
                        second.signal_pdf(second.data.as_dict(), {'NR.x': 1.0}))


def test_sessions_with_efficiency_or_veto_do_not_reuse_unweighted_template():
    model = _model()
    data = model.generate_phase_space(32, seed=403)
    unweighted = FitSession(model, data).signal_cache.normalization_matrix_fixed
    for kwargs in (
        {'efficiency': lambda d: 0.25 * jnp.ones_like(d['s12'])},
        {'veto': lambda d: d['s12'] > 1.0},
    ):
        session = FitSession(model, data, **kwargs)
        weighted = session.signal_cache.normalization_matrix_fixed
        assert not jnp.allclose(weighted, unweighted)
        assert jnp.allclose(session._cached_signal_density({'NR.x': 1.0}),
                            session.signal_pdf(data.as_dict(), {'NR.x': 1.0}),
                            atol=1e-12)


def _toy_session(n_events=400, seed=7):
    model = _model()
    data = generate_signal_toy(model, n_events, seed=seed)
    return FitSession(model, data)


def _toy_result(values=None):
    return SimpleNamespace(values=values or {"NR.x": 1.0}, fixed=None)


def test_goodness_of_fit_projection_default_free_parameters_counts_only_floating():
    session = _toy_session()
    result = _toy_result()
    gof = session.goodness_of_fit_projection(result, "s13", bins=10)
    assert isinstance(gof, BinnedChi2Result)
    assert gof.n_free_parameters == 1  # only "NR.x" is a floating Parameter
    assert gof.n_bins <= 10
    assert np.isfinite(gof.chi2)
    assert 0.0 <= gof.p_value_min <= gof.p_value_max <= 1.0
    assert np.isclose(np.sum(gof.observed), session.data.size)


def test_goodness_of_fit_projection_honors_explicit_free_parameters():
    session = _toy_session()
    gof = session.goodness_of_fit_projection(
        _toy_result(), "s13", bins=10, n_free_parameters=0
    )
    assert gof.n_free_parameters == 0
    assert gof.dof_max - gof.dof_min == 0


def test_goodness_of_fit_chi2_2d_matches_total_events():
    session = _toy_session()
    gof = session.goodness_of_fit_chi2(_toy_result(), bins=6)
    assert isinstance(gof, BinnedChi2Result)
    assert gof.pulls.ndim == 2
    assert np.isclose(np.sum(gof.observed), session.data.size)


def test_goodness_of_fit_chi2_square_dalitz_requires_mother_mass_and_masses():
    session = _toy_session()

    with pytest.raises(ValueError, match="square_dalitz"):
        session.goodness_of_fit_chi2(_toy_result(), square_dalitz=True)


def test_point_to_point_dissimilarity_returns_bounded_p_value():
    session = _toy_session(n_events=300)
    result = session.point_to_point_dissimilarity(
        _toy_result(), mc_size=600, n_permutations=40
    )
    assert isinstance(result, PointToPointResult)
    assert result.n_data == session.data.size
    assert result.n_reference == 600
    assert 0.0 <= result.p_value <= 1.0


def test_point_to_point_dissimilarity_respects_max_total_events_guard():
    session = _toy_session(n_events=300)

    with pytest.raises(ValueError, match="max_total_events"):
        session.point_to_point_dissimilarity(
            _toy_result(), mc_size=10_000, max_total_events=1_000
        )


def test_plot_projection_show_pulls_returns_shared_axes_pair():
    import matplotlib.pyplot as plt

    session = _toy_session()
    ax, ax_pulls = session.plot_projection(
        _toy_result(), "s13", bins=10, show_pulls=True
    )
    assert ax.get_shared_x_axes().joined(ax, ax_pulls)
    assert ax_pulls.get_xlabel()
    plt.close("all")


def test_plot_projection_default_return_unchanged_by_show_pulls_option():
    import matplotlib.pyplot as plt

    session = _toy_session()
    ax = session.plot_projection(_toy_result(), "s13", bins=10)
    assert not isinstance(ax, tuple)
    plt.close("all")


def test_plot_projection_show_pulls_rejects_explicit_ax():
    import matplotlib.pyplot as plt

    session = _toy_session()
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="show_pulls=True"):
        session.plot_projection(_toy_result(), "s13", show_pulls=True, ax=ax)
    plt.close("all")
