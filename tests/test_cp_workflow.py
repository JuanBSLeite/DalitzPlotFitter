import jax.numpy as jnp

from dalitzplotfitter import (
    CPBackgroundSpec,
    CPFitSession,
    CPRealImag,
    DecayChannel,
    DecayModel,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
)


def _models():
    x = Parameter.coefficient("NR.x", 1.0, bounds=(0.2, 2.0), owner="NR")
    dx = Parameter.coefficient("NR.dx", 0.1, bounds=(-0.5, 0.5), owner="NR")
    cp = CPRealImag(x, 0.0, dx, 0.0)
    plus = DecayModel(
        DecayChannel("B+", ("K+", "pi+", "pi-")),
        [NonResonant(cp.for_charge(+1))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
        normalization_pair=(0, 2),
    )
    minus = DecayModel(
        DecayChannel("B-", ("K-", "pi-", "pi+")),
        [NonResonant(cp.for_charge(-1))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
        normalization_pair=(0, 2),
    )
    return plus, minus


def _data(offset=0.0):
    return PhaseSpaceSample(
        s12=jnp.asarray([2.0 + offset, 2.3 + offset]),
        s13=jnp.asarray([1.2 + offset, 1.5 + offset]),
        s23=jnp.asarray([4.0 - offset, 3.7 - offset]),
        weights=jnp.ones(2),
    )


def test_cp_fit_session_collects_shared_parameters_once():
    plus, minus = _models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    assert [p.name for p in session.parameters] == ["NR.x", "NR.dx"]
    value = session.objective({"NR.x": 1.0, "NR.dx": 0.1})
    assert jnp.isfinite(value)


def test_cp_fit_session_automatically_builds_joint_background():
    plus, minus = _models()
    fraction = Parameter("signal_fraction", 0.8, bounds=(0.0, 1.0))
    background = CPBackgroundSpec(
        "comb",
        lambda d: jnp.ones_like(d["s12"]),
    )
    session = CPFitSession(
        plus,
        minus,
        _data(),
        _data(0.02),
        backgrounds=(background,),
        signal_fraction=fraction,
    )
    category = session.background_categories[0]
    expected_plus = jnp.mean(plus.normalization_sample.weights)
    expected_minus = jnp.mean(minus.normalization_sample.weights)
    assert jnp.allclose(category.plus_normalization, expected_plus)
    assert jnp.allclose(category.minus_normalization, expected_minus)
    assert jnp.isfinite(
        session.objective({"NR.x": 1.0, "NR.dx": 0.1, "signal_fraction": 0.8})
    )


def test_cp_fit_session_symmetric_efficiency_is_convenient():
    plus, minus = _models()
    efficiency = lambda d: 0.7 + 0.0 * d["s12"]
    session = CPFitSession(plus, minus, _data(), _data(0.02)).with_efficiency(efficiency)
    assert session.plus_efficiency is efficiency
    assert session.minus_efficiency is efficiency
    assert jnp.allclose(session.plus_acceptance_data, 0.7)
    assert jnp.allclose(session.minus_acceptance_data, 0.7)


def test_cp_projection_weights_preserve_joint_event_count():
    plus, minus = _models()
    session = CPFitSession(plus, minus, _data(), _data(0.02))
    values = {"NR.x": 1.0, "NR.dx": 0.1}
    plus_components = session._projection_components(values, "plus")
    minus_components = session._projection_components(values, "minus")
    total = jnp.sum(jnp.asarray(plus_components[0][2])) + jnp.sum(jnp.asarray(minus_components[0][2]))
    assert jnp.allclose(total, session.plus_data.size + session.minus_data.size, rtol=1e-6)
