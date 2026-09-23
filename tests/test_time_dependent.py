from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from dalitzplotfitter import (
    AmplitudeComponent,
    NeutralMesonMixing,
    Parameter,
    PreparedAmplitudeCache,
    RealImag,
    TimeDependentDalitzNLL,
)


def quadrature(n=120, low=0.0, high=6.0):
    x, w = np.polynomial.legendre.leggauss(n)
    return (low + high) / 2 + (high - low) * x / 2, (high - low) * w / 2


class Shape:
    def __init__(self, reverse=False):
        self.reverse = reverse

    def __call__(self, data, parameters=None):
        z = data["z"]
        if self.reverse:
            z = 1 - z
        slope = 0.7 if parameters is None else parameters.get("slope", 0.7)
        return 1 + slope * z + 1j * z * z


def make_nll(times, tags, z=None, dynamic=False, chunk=None, **kwargs):
    if z is None:
        z = np.linspace(0.1, 0.9, len(times))
    zn, wn = quadrature(35, 0, 1)
    params = (Parameter.dynamics("slope", 0.7, owner="a"),) if dynamic else ()
    cache = PreparedAmplitudeCache.prepare(
        (
            AmplitudeComponent("a", Shape(), RealImag(1, 0)),
            AmplitudeComponent("b", Shape(True), RealImag(0.8, 0.3)),
        ),
        data={"z": jnp.asarray(z)},
        normalization_data={"z": jnp.asarray(zn)},
        normalization_weights=jnp.asarray(wn * len(wn)),
        parameters=params,
        normalize_components=False,
        normalization_chunk_size=chunk or 100_000,
    )
    mixing = kwargs.pop("mixing", NeutralMesonMixing(0.0056, 0.003, 0.4103, 0.9, -0.1))
    return TimeDependentDalitzNLL(cache, 1, times, tags, mixing, **kwargs)


def belle_rate(t, a, b, x, y, tau, r):
    u = t / tau
    c = r * b * np.conj(a)
    return (
        np.exp(-u)
        / 2
        * (
            (abs(a) ** 2 + abs(r * b) ** 2) * np.cosh(y * u)
            + (abs(a) ** 2 - abs(r * b) ** 2) * np.cos(x * u)
            + 2 * c.real * np.sinh(y * u)
            - 2 * c.imag * np.sin(x * u)
        )
    )


@pytest.mark.parametrize("x,y", [(0.0056, 0.003), (0.0, 0.0), (-0.2, 0.15)])
def test_belle_equations_and_exact_integrals(x, y):
    from dalitzplotfitter.likelihood.time_dependent import _rate

    mix = NeutralMesonMixing(x, y, 0.4103)
    a, b, r = 0.4 + 0.8j, 1.1 - 0.2j, 0.9 * np.exp(-0.1j)
    ts = np.array([0.0, 0.12, 0.9, 2.0])
    expected = [belle_rate(t, a, b, x, y, 0.4103, r) for t in ts]
    actual = _rate(mix.basis(ts, {}), abs(a) ** 2, abs(r * b) ** 2, r * np.conj(a) * b)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    for limits in [(0.0, np.inf), (0.2, 1.7)]:
        expected = quad(lambda t: belle_rate(t, a, b, x, y, 0.4103, r), *limits)[0]
        actual = _rate(
            mix.integrals(limits, {}), abs(a) ** 2, abs(r * b) ** 2, r * np.conj(a) * b
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-12)


@pytest.mark.parametrize("tag", [1, -1])
@pytest.mark.parametrize("wrong", [0.0, 0.3, 1.0])
def test_joint_time_dalitz_normalization(tag, wrong):
    z, wz = quadrature(35, 0, 1)
    t, wt = quadrature(80, 0.15, 3.0)
    zz, tt = np.meshgrid(z, t)
    nll = make_nll(
        tt.ravel(),
        np.full(tt.size, tag),
        zz.ravel(),
        wrong_tag=wrong,
        time_range=(0.15, 3.0),
    )
    pdf = np.asarray(nll.densities({})).reshape(tt.shape)
    np.testing.assert_allclose(wt @ pdf @ wz, 1, atol=2e-12)


def test_zero_mixing_factorizes_and_wrong_tag_swaps():
    times = np.array([0.1, 0.5, 1.0])
    nll = make_nll(times, np.ones(3), mixing=NeutralMesonMixing(tau=0.4))
    amplitudes, overlap = nll.cache.coherent_groups({}, np.eye(2))
    expected = (
        abs(amplitudes[:, 0]) ** 2 / overlap[0, 0].real * np.exp(-times / 0.4) / 0.4
    )
    np.testing.assert_allclose(nll.densities({}), expected, rtol=1e-12)
    np.testing.assert_allclose(
        replace(nll, wrong_tag=1.0).densities({}),
        replace(nll, tags=-np.ones(3)).densities({}),
    )


def test_dalitz_integrated_time_pdf_reduces_to_exponential_at_zero_mixing():
    # At x=y=0, mixing.basis gives |g-|^2=0 identically (no oscillation), so
    # both tags' Dalitz-integrated curves must reduce to plain exp(-t/tau)/tau
    # regardless of the coherent overlap (ia, ib, cross) or q/p.
    nll = make_nll(
        np.array([0.1, 0.5, 1.0]), np.array([1, -1, 1]),
        mixing=NeutralMesonMixing(0.0, 0.0, 0.7, 1.3, 0.9),
        wrong_tag=0.2,
    )
    times = np.array([0.0, 0.2, 1.5, 4.0])
    p_plus, p_minus = (np.asarray(v) for v in nll.dalitz_integrated_time_pdf(times, {}))
    expected = np.exp(-times / 0.7) / 0.7
    np.testing.assert_allclose(p_plus, expected, rtol=1e-12)
    np.testing.assert_allclose(p_minus, expected, rtol=1e-12)


@pytest.mark.parametrize("wrong", [0.0, 0.25, 0.6])
def test_dalitz_integrated_time_pdf_integrates_to_one_over_time_range(wrong):
    nll = make_nll(
        np.array([0.1, 0.5, 1.0]), np.array([1, -1, 1]),
        mixing=NeutralMesonMixing(0.01, 0.02, 0.4103, 0.9, 0.3),
        wrong_tag=wrong, time_range=(0.0, 5.0),
    )

    def curve(t, index):
        pdf = nll.dalitz_integrated_time_pdf(np.array([t]), {})
        return float(np.asarray(pdf[index])[0])

    total_plus = quad(lambda t: curve(t, 0), 0.0, 5.0)[0]
    total_minus = quad(lambda t: curve(t, 1), 0.0, 5.0)[0]
    np.testing.assert_allclose([total_plus, total_minus], [1.0, 1.0], atol=1e-8)


def test_dalitz_integrated_time_pdf_matches_manual_overlap_combination():
    nll = make_nll(
        np.array([0.1, 0.5, 1.0]), np.array([1, -1, 1]),
        mixing=NeutralMesonMixing(0.015, -0.03, 0.55, 0.85, -0.4),
        wrong_tag=0.15, time_range=(0.0, 4.0),
    )
    times = np.array([0.0, 0.3, 1.2, 3.0])
    _, ia, ib, cross, ratio = nll._overlap_and_ratio({})
    from dalitzplotfitter.likelihood.time_dependent import _rate

    basis = nll.mixing.basis(times, {})
    integral = nll.mixing.integrals(nll.time_range, {})
    raw_plus = _rate(basis, ia, abs(ratio) ** 2 * ib, ratio * cross)
    raw_minus = _rate(basis, ib, ia / abs(ratio) ** 2, np.conj(cross) / ratio)
    norm_plus = _rate(integral, ia, abs(ratio) ** 2 * ib, ratio * cross)
    norm_minus = _rate(integral, ib, ia / abs(ratio) ** 2, np.conj(cross) / ratio)
    expected_plus = 0.85 * (raw_plus / norm_plus) + 0.15 * (raw_minus / norm_minus)
    expected_minus = 0.85 * (raw_minus / norm_minus) + 0.15 * (raw_plus / norm_plus)

    p_plus, p_minus = nll.dalitz_integrated_time_pdf(times, {})
    np.testing.assert_allclose(p_plus, expected_plus, rtol=1e-12)
    np.testing.assert_allclose(p_minus, expected_minus, rtol=1e-12)


def test_dalitz_integrated_time_pdf_rejects_quadrature_and_event_wise_wrong_tag():
    z, wz = quadrature(35, 0, 1)
    quad_nll = make_nll(
        np.array([0.1, 0.5, 1.0]), np.ones(3), time_nodes=z, time_weights=wz,
    )
    with pytest.raises(ValueError, match="time_nodes=None"):
        quad_nll.dalitz_integrated_time_pdf(np.array([0.5]), {})

    array_wrong_nll = make_nll(
        np.array([0.1, 0.5, 1.0]), np.ones(3), wrong_tag=np.array([0.1, 0.2, 0.3]),
    )
    with pytest.raises(ValueError, match="scalar wrong_tag"):
        array_wrong_nll.dalitz_integrated_time_pdf(np.array([0.5]), {})


@pytest.mark.parametrize("chunk", [None, 11])
def test_dynamic_group_cache_and_gradient(chunk):
    nll = make_nll(
        np.array([0.1, 0.5, 1.0]),
        np.array([1, -1, 1]),
        dynamic=True,
        chunk=chunk,
        mixing=NeutralMesonMixing(Parameter("x", 0.02), Parameter("y", 0.03)),
    )
    values = {"slope": 0.9, "x": 0.02, "y": 0.03}
    amps, overlap = nll.cache.coherent_groups(values, np.eye(2))
    z, w = quadrature(35, 0, 1)
    # Independent direct functions, not the cache reduction.
    f = np.column_stack(
        (
            1 + 0.9 * z + 1j * z * z,
            (0.8 + 0.3j) * (1 + 0.7 * (1 - z) + 1j * (1 - z) ** 2),
        )
    )
    np.testing.assert_allclose(overlap, f.conj().T @ (w[:, None] * f), rtol=1e-12)
    np.testing.assert_allclose(
        amps.sum(axis=1), nll.cache.amplitude(values), rtol=1e-12
    )
    gradient = jax.jit(jax.grad(nll))(values)
    for key in values:
        h = 1e-5
        finite = (
            float(nll({**values, key: values[key] + h}))
            - float(nll({**values, key: values[key] - h}))
        ) / (2 * h)
        np.testing.assert_allclose(gradient[key], finite, rtol=1e-6, atol=1e-8)


def test_gaussian_resolution_acceptance_normalization_and_gradient():
    nodes, weights = quadrature(160, 0, 8.0)
    t, wt = quadrature(100, -0.3, 2.0)
    z, wz = quadrature(25, 0, 1)
    zz, tt = np.meshgrid(z, t)
    nll = make_nll(
        tt.ravel(),
        np.ones(tt.size),
        zz.ravel(),
        time_range=(-0.3, 2.0),
        time_nodes=nodes,
        time_weights=weights,
        sigma_t=0.12,
        time_acceptance=lambda t, p: 1 - jnp.exp(-t / 0.15),
        mixing=NeutralMesonMixing(Parameter("x", 0.03), 0.04),
    )
    pdf = np.asarray(nll.densities({})).reshape(tt.shape)
    np.testing.assert_allclose(wt @ pdf @ wz, 1, atol=3e-10)
    h = 1e-5
    grad = jax.jit(jax.grad(nll))({"x": 0.03})["x"]
    finite = (nll({"x": 0.03 + h}) - nll({"x": 0.03 - h})) / (2 * h)
    np.testing.assert_allclose(grad, finite, rtol=2e-6)


def test_unsmeared_acceptance_and_efficiency():
    t, wt = quadrature(90, 0, 3.0)
    z, wz = quadrature(25, 0, 1)
    zz, tt = np.meshgrid(z, t)
    nll = make_nll(
        tt.ravel(),
        -np.ones(tt.size),
        zz.ravel(),
        time_range=(0, 3.0),
        time_nodes=t,
        time_weights=wt,
        time_acceptance=lambda t, p: t / (t + 0.1),
    )
    np.testing.assert_allclose(
        wt @ np.asarray(nll.densities({})).reshape(tt.shape) @ wz, 1, atol=1e-11
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tags=[0, 1]),
        dict(times=[-0.1, 0.2]),
        dict(wrong_tag=-0.1),
        dict(sigma_t=0.1),
        dict(time_nodes=[1.0]),
        dict(efficiency=[1.0, -1.0]),
        dict(time_range=(1.0, 0.0)),
    ],
)
def test_input_validation(kwargs):
    nll = make_nll(np.array([0.1, 0.2]), np.ones(2))
    with pytest.raises(ValueError):
        replace(nll, **kwargs)


def test_invalid_floating_parameters_reject_objective():
    nll = make_nll(
        np.array([0.1, 0.2]),
        np.ones(2),
        mixing=NeutralMesonMixing(y=Parameter("y", 0.01)),
    )
    assert np.isinf(nll({"y": 1.2}))


def test_gaussian_matches_independent_scipy_convolution():
    from scipy.special import ndtr

    from dalitzplotfitter.likelihood.time_dependent import _rate

    nodes, weights = quadrature(220, 0, 9.0)
    times = np.array([-0.1, 0.25, 1.0])
    nll = make_nll(
        times,
        np.ones(3),
        time_range=(-0.3, 2.0),
        time_nodes=nodes,
        time_weights=weights,
        sigma_t=np.array([0.08, 0.12, 0.2]),
    )
    basis, integrated, _ = nll._time_basis({})
    a, b, r = 0.4 + 0.8j, 1.1 - 0.2j, 0.9 * np.exp(-0.1j)
    actual = _rate(basis, abs(a) ** 2, abs(r * b) ** 2, r * np.conj(a) * b)
    actual_norm = _rate(integrated, abs(a) ** 2, abs(r * b) ** 2, r * np.conj(a) * b)
    for i, (t, sigma) in enumerate(zip(times, nll.sigma_t, strict=True)):

        def rate(u):
            return belle_rate(u, a, b, 0.0056, 0.003, 0.4103, r)

        expected = quad(
            lambda u, t=t, sigma=sigma: (
                rate(u)
                * np.exp(-0.5 * ((t - u) / sigma) ** 2)
                / (np.sqrt(2 * np.pi) * sigma)
            ),
            0,
            9,
            epsabs=1e-12,
        )[0]
        norm = quad(
            lambda u, sigma=sigma: (
                rate(u) * (ndtr((2 - u) / sigma) - ndtr((-0.3 - u) / sigma))
            ),
            0,
            9,
            epsabs=1e-12,
        )[0]
        np.testing.assert_allclose(actual[i], expected, rtol=2e-10)
        np.testing.assert_allclose(actual_norm[i], norm, rtol=2e-10)


def test_flavour_exchange_and_all_mixing_gradients():
    mix = NeutralMesonMixing(
        *(
            Parameter(k, v)
            for k, v in zip(
                ("x", "y", "tau", "q", "phi"),
                (0.03, -0.02, 0.4, 1.15, 0.2),
                strict=True,
            )
        )
    )
    nll = make_nll(np.array([0.1, 0.3, 1.0]), np.array([1, -1, 1]), mixing=mix)
    values = {p.name: p.value for p in mix.parameters}
    grad = jax.jit(jax.grad(nll))(values)
    for k in values:
        h = 1e-5
        finite = (
            nll({**values, k: values[k] + h}) - nll({**values, k: values[k] - h})
        ) / (2 * h)
        np.testing.assert_allclose(grad[k], finite, rtol=1e-6, atol=2e-8)
    swapped = replace(
        nll.cache,
        data_components=nll.cache.data_components[:, ::-1],
        normalization_matrix_fixed=nll.cache.normalization_matrix_fixed[::-1, ::-1],
        components=nll.cache.components[::-1],
    )
    other = replace(
        nll,
        cache=swapped,
        tags=-nll.tags,
        mixing=NeutralMesonMixing(0.03, -0.02, 0.4, 1 / 1.15, -0.2),
    )
    np.testing.assert_allclose(nll.densities(values), other.densities({}), rtol=1e-12)


def test_efficiency_normalization_and_no_fixed_dynamics_reevaluation():
    z, wz = quadrature(25, 0, 1)
    t, wt = quadrature(70, 0, 3)
    zz, tt = np.meshgrid(z, t)
    nll = make_nll(tt.ravel(), np.ones(tt.size), zz.ravel(), time_range=(0, 3))
    old = nll.cache
    cache = PreparedAmplitudeCache.prepare(
        old.components,
        data={"z": jnp.asarray(zz.ravel())},
        normalization_data={"z": jnp.asarray(z)},
        normalization_weights=jnp.asarray(wz * len(z)),
        efficiency_normalization=jnp.asarray(0.2 + z),
        normalize_components=False,
    )
    nll = replace(nll, cache=cache, efficiency=0.2 + zz.ravel())

    # A compact cache must not call the component functions after preparation.
    class Forbidden:
        def __call__(self, *args, **kwargs):
            raise AssertionError("fixed dynamics reevaluated")

    cache = replace(
        cache,
        components=tuple(replace(c, function=Forbidden()) for c in cache.components),
    )
    nll = replace(nll, cache=cache)
    pdf = np.asarray(jax.jit(nll.densities)({})).reshape(tt.shape)
    np.testing.assert_allclose(wt @ pdf @ wz, 1, atol=1e-12)
