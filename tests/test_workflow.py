import jax.numpy as jnp

from dalitzplotfitter import (
    BackgroundSpec,
    DecayChannel,
    DecayModel,
    FitSession,
    GaussianConstraint,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    RealImag,
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
