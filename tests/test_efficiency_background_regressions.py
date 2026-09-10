import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    BackgroundCategory,
    CPBackgroundCategory,
    DecayChannel,
    DecayModel,
    FitSession,
    MultiBackgroundNLL,
    NonResonant,
    Parameter,
    RealImag,
)
from dalitzplotfitter.workflow import _acceptance


def background(**kw):
    return BackgroundCategory("bg", jnp.ones(2), 1.0, **kw)


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_fraction_at_construction_and_under_jit(bad):
    with pytest.raises(ValueError, match="physical"):
        MultiBackgroundNLL(lambda p: jnp.ones(2), (background(),), signal_fraction=bad)
    nll = MultiBackgroundNLL(
        lambda p: jnp.ones(2), (background(),), signal_fraction=Parameter("f", 0.5)
    )
    value, grad = jax.jit(jax.value_and_grad(lambda f: nll({"f": f})))(bad)
    assert np.isinf(value)
    assert np.isfinite(grad)


def test_simplex_and_extended_yields():
    bgs = tuple(
        BackgroundCategory(
            str(i), jnp.ones(2), 1.0, fraction=Parameter(str(i), 0.2) if i < 2 else None
        )
        for i in range(3)
    )
    nll = MultiBackgroundNLL(lambda p: jnp.ones(2), bgs, signal_fraction=0.5)
    assert np.isinf(jax.jit(lambda a: nll({"0": a, "1": a}))(0.8))
    ext = MultiBackgroundNLL(
        lambda p: jnp.ones(2),
        (background(yield_=Parameter("b", 1.0)),),
        extended=True,
        signal_yield=3.0,
    )
    assert np.isinf(jax.jit(lambda b: ext({"b": b}))(-1.0))
    assert np.isfinite(ext({"b": 0.0}))
    with pytest.raises(ValueError, match="physical"):
        MultiBackgroundNLL(
            lambda p: jnp.ones(2),
            (background(yield_=-1.0),),
            extended=True,
            signal_yield=3.0,
        )


@pytest.mark.parametrize("density", [0.0, -1.0, np.nan, np.inf])
def test_invalid_total_density(density):
    nll = MultiBackgroundNLL(lambda p: jnp.full(2, density))
    assert np.isinf(jax.jit(lambda: nll({}))())


def test_background_can_cover_signal_zero():
    nll = MultiBackgroundNLL(
        lambda p: jnp.zeros(2), (background(),), signal_fraction=0.5
    )
    assert np.isclose(nll({}), 2 * np.log(2))


@pytest.mark.parametrize(
    "function",
    [
        lambda d: jnp.ones((3, 1)),
        lambda d: -jnp.ones(3),
        lambda d: jnp.full(3, jnp.nan),
    ],
)
@pytest.mark.parametrize("slot", ["efficiency", "veto"])
def test_acceptance_rejects_invalid_output(function, slot):
    kw = dict(efficiency=None, veto=None, data={"s12": jnp.ones(3)})
    kw[slot] = function
    with pytest.raises(ValueError, match=slot):
        _acceptance(**kw)


def test_scalar_relative_efficiency():
    np.testing.assert_array_equal(
        _acceptance(lambda d: 7.0, None, {"s12": jnp.ones(3)}), [7.0, 7.0, 7.0]
    )


@pytest.mark.parametrize("charge", ["plus", "minus"])
def test_cp_background_one_charge(charge):
    options = dict(
        plus_values=jnp.zeros(2),
        minus_values=jnp.zeros(2),
        plus_normalization=0.0,
        minus_normalization=0.0,
    )
    options[charge + "_values"] = jnp.ones(2)
    options[charge + "_normalization"] = 2.0
    bg = CPBackgroundCategory("bg", **options)
    assert float(getattr(bg, charge + "_probability")) == 1.0
    np.testing.assert_array_equal(getattr(bg, charge + "_density"), [0.5, 0.5])


@pytest.mark.parametrize(
    "minus_values,minus_norm",
    [(jnp.ones(2), 0.0), (jnp.zeros(2), -1.0), (jnp.zeros(2), np.nan)],
)
def test_cp_background_invalid_charge(minus_values, minus_norm):
    with pytest.raises(ValueError):
        CPBackgroundCategory("bg", jnp.ones(2), minus_values, 1.0, minus_norm)
    with pytest.raises(ValueError):
        CPBackgroundCategory("bg", jnp.zeros(2), jnp.zeros(2), 0.0, 0.0)


def test_vetoed_data_have_infinite_nll_and_zero_pdf():
    model = DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
    )
    data = model.generate_phase_space(10, seed=2, include_momenta=False)

    def veto(d):
        return ~jnp.any(d["s12"][:, None] == data.s12[None, :], axis=1)

    session = FitSession(model, data, veto=veto)
    objective = session.objective
    assert np.isinf(jax.jit(lambda: objective({}))())
    np.testing.assert_array_equal(session.signal_pdf(data.as_dict(), {}), np.zeros(10))
    assert np.all(np.isneginf(session.signal_pdf.logpdf(data.as_dict(), {})))


def test_valid_mixture_gradient_and_float32_density():
    signal = jnp.array([0.2, 0.8], dtype=jnp.float32)
    nll = MultiBackgroundNLL(
        lambda p: signal, (background(),), signal_fraction=Parameter("f", 0.5)
    )
    value, grad = jax.jit(jax.value_and_grad(lambda f: nll({"f": f})))(0.4)
    density = 0.4 * np.asarray(signal) + 0.6
    assert np.isclose(value, -np.sum(np.log(density)))
    assert np.isclose(grad, -np.sum((np.asarray(signal) - 1) / density))
