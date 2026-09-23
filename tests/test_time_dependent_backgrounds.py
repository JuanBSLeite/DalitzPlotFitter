from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dalitzplotfitter import (
    BackgroundCategory,
    DecayChannel,
    DecayModel,
    NeutralMesonMixing,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    RealImag,
    TimeDependentBackgroundCategory,
    TimeDependentFitSession,
)


def _session(n=12, **kwargs):
    model = DecayModel(
        DecayChannel("D0", ("K(S)0", "pi+", "pi-")),
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=8,
    )
    sample = model.normalization_sample
    data = PhaseSpaceSample(
        s12=sample.s12[:n],
        s13=sample.s13[:n],
        s23=sample.s23[:n],
        weights=jnp.ones(n),
    )
    return TimeDependentFitSession(
        model,
        data,
        jnp.linspace(0.1, 1.8, n),
        jnp.where(jnp.arange(n) % 3, 1, -1),
        NeutralMesonMixing(0.0, 0.0, Parameter("tau", 0.4103, bounds=(0.1, 2.0))),
        time_range=(0, 2),
        **kwargs,
    )


def _time(data, parameters):
    rate = parameters.get("rate", 1.4)
    t = data["t"]
    return jnp.where(
        (t >= 0) & (t <= 2), rate * jnp.exp(-rate * t) / -jnp.expm1(-2 * rate), 0
    )


def _shape(data):
    return data["s12"] * jnp.where(data["tag"] == 1, 2.0, 3.0)


def _multi(**kwargs):
    session = _session(**kwargs)
    session = session.with_background(
        "comb",
        _shape,
        time_pdf=_time,
        fraction=Parameter("f_comb", 0.3, bounds=(0, 1)),
        parameters=(Parameter("rate", 1.4, bounds=(0.1, 5)),),
    ).with_background("other", lambda d: jnp.ones_like(d["s12"]), time_pdf=_time)
    return replace(session, signal_fraction=Parameter("f_sig", 0.7, bounds=(0, 1)))


def test_conditional_mixture_matches_manual_and_jit_gradients():
    session = _multi()
    objective = session.base_objective
    values = {"tau": 0.5, "rate": 2.0, "f_sig": 0.6, "f_comb": 0.2}
    signal = session.signal_objective.densities(values)
    first, second = [c.density(values) for c in session.background_categories]
    expected = 0.6 * signal + 0.4 * (0.2 * first + 0.8 * second)
    np.testing.assert_allclose(objective.density(values), expected)
    np.testing.assert_allclose(objective(values), -np.log(expected).sum())
    grads = jax.jit(jax.grad(objective))(values)
    assert {p.name for p in session.parameters} == set(values)
    for name, point in values.items():
        step = 1e-5
        finite = (
            objective({**values, name: point + step})
            - objective({**values, name: point - step})
        ) / (2 * step)
        np.testing.assert_allclose(grads[name], finite, rtol=1e-5, atol=1e-7)
    assert abs(float(grads["rate"])) > 0.01


@pytest.mark.parametrize("extended", [False, True])
def test_joint_normalization_with_multiple_backgrounds(extended):
    session = _multi()
    norm = session.model.normalization_sample
    nodes, wt = np.polynomial.legendre.leggauss(45)
    t = nodes + 1
    idx = np.tile(np.arange(norm.size), len(t))
    data = PhaseSpaceSample(
        s12=norm.s12[idx],
        s13=norm.s13[idx],
        s23=norm.s23[idx],
        weights=jnp.ones(len(idx)),
    )
    integral = 0.0
    for tag in (1, -1):
        selected = replace(
            session,
            data=data,
            times=np.repeat(t, norm.size),
            tags=np.full(len(idx), tag),
        )
        if extended:
            selected = replace(
                selected,
                extended=True,
                signal_fraction=None,
                signal_yield=70.0,
                signal_tag_fraction=0.8,
                backgrounds=tuple(
                    replace(b, fraction=None, yield_=y, tag_fraction=f)
                    for b, y, f in zip(
                        selected.backgrounds, [10.0, 20.0], [0.2, 0.6], strict=True
                    )
                ),
            )
        density = np.asarray(selected.base_objective.density({})).reshape(
            len(t), norm.size
        )
        area = wt @ density @ np.asarray(norm.weights) / norm.size
        if not extended:
            np.testing.assert_allclose(area, 1.0, atol=1e-12)
        integral += area
    np.testing.assert_allclose(integral, 100.0 if extended else 2.0, atol=1e-10)


def test_extended_poisson_term_and_tag_splits():
    session = _session(extended=True, signal_yield=7.0, signal_tag_fraction=0.8)
    session = session.with_background(
        "comb", _shape, time_pdf=_time, yield_=3.0, tag_fraction=0.1
    )
    objective = session.base_objective
    tag = np.asarray(session.tags)
    expected = 7 * np.where(tag == 1, 0.8, 0.2) * session.signal_objective.densities(
        {}
    ) + 3 * np.where(tag == 1, 0.1, 0.9) * session.background_categories[0].density({})
    np.testing.assert_allclose(objective({}), 10 - np.log(expected).sum())
    np.testing.assert_allclose(session._projection_scales({}, 1)[0], 5.6)
    np.testing.assert_allclose(session._projection_scales({}, 1)[1], [0.3])
    plus, bp = session._projection_scales({}, 1)
    minus, bm = session._projection_scales({}, -1)
    assert plus + minus + sum(bp) + sum(bm) == pytest.approx(10)


def test_extended_fit_recovers_total_event_count():
    session = _session(
        extended=True, signal_yield=Parameter("ns", 8.0, bounds=(0, 100))
    )
    # Identical signal/background shapes isolate the Poisson-yield check.
    signal = session.signal_objective
    category = TimeDependentBackgroundCategory("same", signal.densities, yield_=2.0)
    session = replace(
        session, backgrounds=(category,), mixing=NeutralMesonMixing(0, 0, 0.4103)
    )
    result = session.fit(hesse=True)
    assert result.valid
    assert result.values["ns"] == pytest.approx(session.data.size - 2, abs=0.01)


def test_invalid_mixing_cannot_be_hidden_by_background():
    session = _multi()
    objective = session.base_objective
    f = jax.jit(jax.value_and_grad(lambda tau: objective({"tau": tau})))
    value, gradient = f(0.0)
    assert np.isinf(value)
    assert np.isfinite(gradient)


@pytest.mark.parametrize(
    "change",
    [
        {"signal_fraction": -0.1},
        {"signal_fraction": 1.1},
        {"extended": True},
        {"signal_yield": 2.0},
        {"signal_tag_fraction": 0.5},
    ],
)
def test_invalid_mixture_configuration_rejected(change):
    with pytest.raises(ValueError):
        _ = replace(_multi(), **change).base_objective


def test_duplicate_background_names_and_fraction_remainder_rejected():
    session = _multi()
    with pytest.raises(ValueError, match="unique"):
        _ = replace(session, backgrounds=(session.backgrounds[0],) * 2).base_objective
    with pytest.raises(ValueError, match="remainder"):
        _ = replace(
            session,
            backgrounds=tuple(replace(b, fraction=0.4) for b in session.backgrounds),
        ).base_objective


def test_precomputed_and_dynamic_correlated_categories():
    session = _session(signal_fraction=0.6)
    signal = session.signal_objective
    category = TimeDependentBackgroundCategory(
        "random_tag",
        replace(signal, wrong_tag=0.5).densities,
    )
    session = replace(session, backgrounds=(category,))
    expected = 0.6 * signal.densities({}) + 0.4 * replace(
        signal, wrong_tag=0.5
    ).densities({})
    np.testing.assert_allclose(session.base_objective.density({}), expected)
    fixed = BackgroundCategory("fixed", np.asarray(expected), 1.0)
    session = replace(session, backgrounds=(fixed,))
    np.testing.assert_allclose(session.background_categories[0].density, expected)
    with pytest.raises(ValueError, match="dalitz_density"):
        session.plot_projection(
            SimpleNamespace(values={"tau": 0.4103}), projection_size=100
        )
    plt.close("all")


def test_veto_applied_to_background_but_not_signal_efficiency():
    session = _multi(veto=lambda d: d["s12"] > 0.9)
    category = session.background_categories[0]
    data = session.model.normalization_sample
    d = dict(data.as_dict(), tag=jnp.ones(data.size))
    pdf = category.dalitz_density(d, {})
    assert np.all(np.asarray(pdf)[np.asarray(data.s12) <= 0.9] == 0)
    np.testing.assert_allclose(jnp.mean(data.weights * pdf), 1.0)
    changed = session.with_efficiency(lambda d: 0.2 + 0.1 * d["s12"])
    np.testing.assert_allclose(
        category.density({}), changed.background_categories[0].density({})
    )


def test_projections_include_all_backgrounds_and_total():
    session = _multi()
    values = {p.name: p.value for p in session.parameters}
    result = SimpleNamespace(values=values)
    axes = session.plot_projection(
        result, bins=8, range=(0.0, 4.0), projection_size=400
    )
    for tag, ax in zip((1, -1), axes, strict=True):
        components = {
            p.get_label(): np.asarray(p.get_data().values) for p in ax.patches
        }
        label = "D0" if tag == 1 else "D0bar"
        total = components[f"fit ({label})"]
        np.testing.assert_allclose(
            total, sum(v for k, v in components.items() if not k.startswith("fit"))
        )
        np.testing.assert_allclose(total.sum(), np.sum(np.asarray(session.tags) == tag))
    ax = session.plot_time_projection(result, bins=8)
    curves = {
        line.get_label(): line.get_ydata()
        for line in ax.lines
        if line.get_label()[0] != "_"
    }
    for label in ("D0", "D0bar"):
        np.testing.assert_allclose(
            curves[f"fit ({label})"],
            sum(curves[f"{name} ({label})"] for name in ("signal", "comb", "other")),
        )
    plt.close("all")


def test_floating_tag_probabilities_collected_and_differentiated():
    session = _session(
        extended=True,
        signal_yield=8.0,
        signal_tag_fraction=Parameter("p_sig", 0.7, bounds=(0, 1)),
    ).with_background(
        "comb",
        _shape,
        time_pdf=_time,
        yield_=4.0,
        tag_fraction=Parameter("p_comb", 0.4, bounds=(0, 1)),
    )
    values = {p.name: p.value for p in session.parameters}
    objective = session.base_objective
    gradients = jax.jit(jax.grad(objective))(values)
    for name in ("p_sig", "p_comb"):
        step = 1e-5
        expected = (
            objective({**values, name: values[name] + step})
            - objective({**values, name: values[name] - step})
        ) / (2 * step)
        np.testing.assert_allclose(gradients[name], expected, atol=1e-7)
    assert np.isinf(objective({**values, "p_comb": 1.1}))


def test_fixed_background_shape_is_cached_across_objective_calls():
    calls = []

    def shape(data):
        calls.append(len(data["s12"]))
        return _shape(data)

    session = _session(signal_fraction=0.8).with_background(
        "comb", shape, time_pdf=_time
    )
    objective = session.base_objective
    initial = len(calls)
    vg = jax.jit(jax.value_and_grad(lambda rate: objective({"rate": rate})))
    vg(1.4)
    vg(2.0)
    assert len(calls) == initial


def test_general_correlated_background_has_time_dependent_dalitz_shape():
    session = _session(signal_fraction=0.6)
    sample = session.model.normalization_sample
    volume = jnp.mean(sample.weights)
    data = session._event_data()
    eps = Parameter("correlation", 0.5, bounds=(-1, 1))

    def joint(values):
        # Integral over t in [0,2] is 1/volume at every Dalitz point.
        return (1 + eps.resolve(values) * data["s12"] / 4 * (data["t"] - 1)) / (
            2 * volume
        )

    category = TimeDependentBackgroundCategory("correlated", joint, parameters=(eps,))
    session = replace(session, backgrounds=(category,))
    assert "correlation" in {p.name for p in session.parameters}
    objective = session.base_objective
    derivative = jax.grad(lambda e: objective({"correlation": e}))(0.5)
    assert abs(float(derivative)) > 1e-4
    expected = 0.6 * session.signal_objective.densities({}) + 0.4 * joint({})
    np.testing.assert_allclose(objective.density({}), expected)


@pytest.mark.parametrize("invalid", [lambda p: -jnp.ones(12), lambda p: jnp.ones(3)])
def test_invalid_dynamic_background_pdf_rejected(invalid):
    category = TimeDependentBackgroundCategory("invalid", invalid)
    session = _session(signal_fraction=0.99, backgrounds=(category,))
    if invalid({}).shape != (12,):
        with pytest.raises(ValueError, match="shape"):
            session.objective({})
    else:
        assert np.isinf(session.objective({}))


def test_floating_dalitz_shape_cannot_silently_enter_fixed_cache():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Shape:
        slope: object

        def __call__(self, data):
            return 1 + self.slope.value * data["s12"]

    session = _session(signal_fraction=0.8).with_background(
        "comb",
        Shape(Parameter("slope", 0.1)),
        time_pdf=_time,
    )
    with pytest.raises(ValueError, match="must be fixed"):
        _ = session.base_objective
