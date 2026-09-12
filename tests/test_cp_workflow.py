from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    BinnedChi2Result,
    CPBackgroundSpec,
    CPFitSession,
    CPRealImag,
    DecayChannel,
    DecayModel,
    NonResonant,
    Parameter,
    PhaseSpaceSample,
    PointToPointResult,
    generate_signal_toy,
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


def _toy_cp_session(n_events=300, seed=11):
    plus, minus = _models()
    plus_data = generate_signal_toy(plus, n_events, seed=seed)
    minus_data = generate_signal_toy(minus, n_events, seed=seed + 1)
    return CPFitSession(plus, minus, plus_data, minus_data)


def _toy_cp_result():
    return SimpleNamespace(values={"NR.x": 1.0, "NR.dx": 0.1}, fixed=None)


def test_cp_goodness_of_fit_projection_returns_both_charges_by_default():
    session = _toy_cp_session()
    result = _toy_cp_result()
    gof = session.goodness_of_fit_projection(result, "s13", bins=8)
    assert set(gof) == {"plus", "minus"}
    for charge_result in gof.values():
        assert isinstance(charge_result, BinnedChi2Result)
        assert charge_result.n_free_parameters == 2
        assert np.isfinite(charge_result.chi2)


def test_cp_goodness_of_fit_projection_single_charge():
    session = _toy_cp_session()
    result = _toy_cp_result()
    plus_only = session.goodness_of_fit_projection(result, "s13", bins=8, charge="plus")
    assert isinstance(plus_only, BinnedChi2Result)
    with pytest.raises(ValueError, match="charge"):
        session.goodness_of_fit_projection(result, "s13", charge="neutral")


def test_cp_goodness_of_fit_chi2_2d_returns_both_charges():
    session = _toy_cp_session()
    result = _toy_cp_result()
    gof = session.goodness_of_fit_chi2(result, bins=5)
    assert set(gof) == {"plus", "minus"}
    assert gof["plus"].pulls.ndim == 2


def test_cp_point_to_point_dissimilarity_requires_charge():
    session = _toy_cp_session(n_events=200)
    result = _toy_cp_result()
    with pytest.raises(TypeError):
        session.point_to_point_dissimilarity(result)
    with pytest.raises(ValueError, match="charge"):
        session.point_to_point_dissimilarity(result, charge="neutral")

    plus_result = session.point_to_point_dissimilarity(
        result, charge="plus", mc_size=400, n_permutations=30
    )
    assert isinstance(plus_result, PointToPointResult)
    assert plus_result.n_data == session.plus_data.size
    assert 0.0 <= plus_result.p_value <= 1.0


def test_cp_plot_projection_show_pulls_returns_2x2_grid():
    import matplotlib.pyplot as plt

    session = _toy_cp_session()
    grid = session.plot_projection(
        _toy_cp_result(), "s13", bins=10, show_pulls=True
    )
    assert grid.shape == (2, 2)
    assert grid[0, 0].get_shared_x_axes().joined(grid[0, 0], grid[1, 0])
    assert grid[0, 1].get_shared_x_axes().joined(grid[0, 1], grid[1, 1])
    plt.close("all")


def test_cp_plot_projection_default_return_unchanged_by_show_pulls_option():
    import matplotlib.pyplot as plt

    session = _toy_cp_session()
    axes = session.plot_projection(_toy_cp_result(), "s13", bins=10)
    assert axes.shape == (2,)
    plt.close("all")


def test_cp_plot_projection_show_pulls_rejects_explicit_axes():
    import matplotlib.pyplot as plt

    session = _toy_cp_session()
    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="show_pulls=True"):
        session.plot_projection(_toy_cp_result(), "s13", show_pulls=True, axes=axes)
    plt.close("all")
