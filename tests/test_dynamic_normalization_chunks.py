"""Chunked dynamics must preserve global scales, interference and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    QMI,
    CPRealImag,
    DecayChannel,
    DecayModel,
    GounarisSakurai,
    Parameter,
    RealImag,
    RelativisticBreitWigner,
    Resonance,
    ZemachP,
)
from dalitzplotfitter.amplitude import PreparedAmplitudeCache
from dalitzplotfitter.likelihood import CPJointNLL


def make_model(
    chunk_size,
    *,
    qmi=None,
    all_dynamic=False,
    normalize=True,
    charge=1,
    dynamics_microbatch_size=20_000,
):
    mass = Parameter.dynamics("rho.mass", 0.775, owner="rho")
    width = Parameter.dynamics("rho.width", 0.149, owner="rho")
    shape = Parameter.dynamics("s.shape", 0.6, owner="s")
    coefficient = Parameter.coefficient("rho.x", 0.8, owner="rho")
    cp = CPRealImag(coefficient, 0.1, 0.07, -0.03)
    components = [
        Resonance(
            "rho",
            (2, 0),
            cp.for_charge(charge),
            spin=1,
            mass=mass,
            width=width,
            lineshape=GounarisSakurai(),
            angular=ZemachP(),
        ),
        Resonance(
            "s",
            (2, 0),
            RealImag(0.4, -0.2),
            spin=0,
            mass=1.0 if qmi else shape,
            width=0.4,
            lineshape=QMI(
                knots=(0.28, 0.5, 0.9, 1.7),
                real_parts=(1.0, shape, -0.3, 0.5),
                imaginary_parts=(0.2, -0.1, 0.4, 0.1),
                interpolation=qmi,
            )
            if qmi
            else RelativisticBreitWigner(),
            normalize_component=False,
        ),
    ]
    if not all_dynamic:
        components.insert(
            1,
            Resonance(
                "fixed",
                (2, 0),
                RealImag(0.3, 0.4),
                spin=2,
                mass=1.25,
                width=0.17,
                normalize_component=True,
            ),
        )
    return DecayModel(
        DecayChannel("D+", ("pi+", "pi+", "pi-")),
        components,
        normalization_method="square-dalitz",
        normalization_resolution=9,
        normalization_pair=(0, 1),
        normalization_chunk_size=chunk_size,
        dynamics_microbatch_size=dynamics_microbatch_size,
        normalize_components=normalize,
    )


def prepare_pair(
    *,
    qmi=None,
    all_dynamic=False,
    normalize=True,
    efficiency=True,
    charge=1,
    dynamics_microbatch_size=20_000,
    chunk_size=17,
):
    full = make_model(
        1000, qmi=qmi, all_dynamic=all_dynamic, normalize=normalize, charge=charge
    )
    chunked = make_model(
        chunk_size,
        qmi=qmi,
        all_dynamic=all_dynamic,
        normalize=normalize,
        charge=charge,
        dynamics_microbatch_size=dynamics_microbatch_size,
    )
    data = full.generate_phase_space(19, seed=381)
    sample = full.normalization_sample
    acceptance = (
        jnp.where(
            jnp.arange(sample.size) % 7 == 0, 0.0, jnp.linspace(0.3, 0.9, sample.size)
        )
        if efficiency
        else None
    )
    return tuple(
        m.prepare_cache(data, sample, efficiency_normalization=acceptance)
        for m in (full, chunked)
    )


def mapping(x):
    return dict(zip(("rho.mass", "rho.width", "s.shape", "rho.x"), x, strict=True))


def nll(cache, x):
    intensity, norm = cache.evaluate(mapping(x))
    return -jnp.log(intensity).sum() + intensity.size * jnp.log(norm)


@pytest.mark.parametrize(
    "all_dynamic,normalize,efficiency",
    [
        (False, True, True),
        (False, False, False),
        (True, True, True),
    ],
)
def test_chunked_mass_width_values_gradients_and_hessian(
    all_dynamic, normalize, efficiency
):
    full, chunked = prepare_pair(
        all_dynamic=all_dynamic, normalize=normalize, efficiency=efficiency
    )
    assert chunked.normalization_chunks is not None  # verifies DecayModel threading
    assert full.normalization_chunks is None
    for point in ([0.775, 0.149, 0.6, 0.8], [0.79, 0.16, 0.62, 0.75]):
        x = jnp.asarray(point)
        a, b = (lambda x: nll(full, x)), (lambda x: nll(chunked, x))
        for expected, actual in zip(
            jax.jit(jax.value_and_grad(a))(x),
            jax.jit(jax.value_and_grad(b))(x),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-10)
        np.testing.assert_allclose(
            jax.jit(jax.jacfwd(jax.grad(a)))(x),
            jax.jit(jax.jacfwd(jax.grad(b)))(x),
            rtol=2e-10,
            atol=1e-9,
        )
        for method in (
            "amplitude",
            "normalization",
            "normalization_matrix",
            "fit_fractions",
        ):
            np.testing.assert_allclose(
                getattr(full, method)(mapping(x)),
                getattr(chunked, method)(mapping(x)),
                rtol=2e-12,
                atol=1e-12,
            )
        for expected, actual in zip(
            full._evaluate_components(mapping(x)),
            chunked._evaluate_components(mapping(x)),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-12)
        _, full_dynamic_norm = full._evaluate_dynamic_components(mapping(x))
        _, chunked_dynamic_norm = chunked._evaluate_dynamic_components(mapping(x))
        np.testing.assert_allclose(
            chunked._matrix_from_dynamic(chunked_dynamic_norm),
            full._matrix_from_dynamic(full_dynamic_norm),
            rtol=2e-12,
            atol=1e-12,
        )


def test_chunked_dynamics_microbatching_matches_unchunked():
    """Force >1 inner-scan microbatch iteration and check it still matches.

    Every other test in this file uses ``chunk_size=17`` against the default
    ``dynamics_microbatch_size=20_000``, so the inner scan has a single
    iteration there. Setting the public option below the macro chunk size
    exercises the padded multi-iteration path directly.
    """
    full, chunked = prepare_pair(
        all_dynamic=True,
        normalize=True,
        efficiency=True,
        dynamics_microbatch_size=5,
        chunk_size=1000,
    )
    assert chunked.normalization_chunks is not None
    assert chunked.dynamics_microbatch_size == 5
    # The 81-point sample fits in the 1000-point macroblock, but the public
    # 5-point microbatch limit must still activate the chunked path.
    assert chunked.normalization_chunks[1].shape[1] == 81
    x = jnp.asarray([0.79, 0.16, 0.62, 0.75])
    a, b = (lambda x: nll(full, x)), (lambda x: nll(chunked, x))
    for expected, actual in zip(
        jax.jit(jax.value_and_grad(a))(x),
        jax.jit(jax.value_and_grad(b))(x),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-10)
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(jax.grad(a)))(x),
        jax.jit(jax.jacfwd(jax.grad(b)))(x),
        rtol=2e-10,
        atol=1e-9,
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic", "hermite", "natural"])
def test_chunked_qmi_preparation_and_second_derivatives(interpolation):
    # QMI's prepared order/starts/ends are valid only for their exact block.
    # When the requested chunk is larger than the AD microbatch bound, QMI must
    # therefore be prepared directly in smaller blocks rather than reshaped
    # after preparation.
    full, chunked = prepare_pair(
        qmi=interpolation,
        dynamics_microbatch_size=5,
    )
    assert chunked.normalization_chunks[1].shape[1] == 5
    x = jnp.asarray([0.79, 0.16, 0.62, 0.75])
    a, b = (lambda x: nll(full, x)), (lambda x: nll(chunked, x))
    np.testing.assert_allclose(
        jax.jit(jax.grad(a))(x), jax.jit(jax.grad(b))(x), rtol=2e-10, atol=1e-9
    )
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(jax.grad(a)))(x),
        jax.jit(jax.jacfwd(jax.grad(b)))(x),
        rtol=2e-9,
        atol=1e-8,
    )
    # The reusable fraction kernel receives chunk arrays as runtime inputs.
    values = mapping(x)
    kernel = chunked._build_fraction_jacobian_kernel(tuple(values))
    actual = kernel(values, chunked._fraction_jacobian_arrays())
    expected = jax.jacrev(lambda v: full.fit_fractions(mapping(v)))(x)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-10)


def test_chunked_cp_joint_normalization_and_derivatives():
    plus = prepare_pair(charge=1)
    minus = prepare_pair(charge=-1)
    full, chunked = (CPJointNLL(p, m) for p, m in zip(plus, minus, strict=True))
    x = jnp.asarray([0.79, 0.16, 0.62, 0.75])
    a, b = (lambda x: full(mapping(x))), (lambda x: chunked(mapping(x)))
    np.testing.assert_allclose(
        jax.jit(jax.grad(a))(x), jax.jit(jax.grad(b))(x), rtol=2e-11, atol=1e-10
    )
    np.testing.assert_allclose(a(x), b(x), rtol=1e-12)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_dynamic_chunk_size_must_be_positive(chunk_size):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        PreparedAmplitudeCache.prepare(
            (object(),),
            data={},
            normalization_data={},
            normalization_weights=jnp.ones(3),
            normalization_chunk_size=chunk_size,
        )


@pytest.mark.parametrize("microbatch_size", [0, -1, True, 1.5])
def test_dynamic_microbatch_size_must_be_a_positive_integer(microbatch_size):
    with pytest.raises(ValueError, match="dynamics_microbatch_size"):
        PreparedAmplitudeCache.prepare(
            (object(),),
            data={},
            normalization_data={},
            normalization_weights=jnp.ones(3),
            dynamics_microbatch_size=microbatch_size,
        )
