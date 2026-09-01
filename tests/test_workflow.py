import jax.numpy as jnp

from dalitzplotfitter import (
    BackgroundSpec,
    FitSession,
    GaussianConstraint,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    RealImag,
    DecayChannel,
    DecayModel,
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
    assert jnp.allclose(constrained.objective({"NR.x": 1.2}) - base.objective({"NR.x": 1.2}), 0.5)


def test_fit_session_projection_weights_reproduce_expected_events():
    session = FitSession(_model(), _data())
    components = session._projection_components({"NR.x": 1.0})
    assert len(components) == 1
    assert jnp.allclose(jnp.sum(jnp.asarray(components[0][2])), session.data.size, rtol=1e-6)
