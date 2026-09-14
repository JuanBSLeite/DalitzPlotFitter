"""Measure, interpolation and charge-yield checks for the Laura analysis adapters."""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from dalitzplotfitter import (
    CPBackgroundCategory,
    MassWindowVeto,
    SquareDalitzGrid,
    square_dalitz_to_invariants,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks/data_analyses"))
from laura_comparison_helpers import (  # noqa: E402
    LauraBackground,
    LauraEfficiency,
    charge_scaled_background,
)

MASS = 5.27934
MASSES = (0.13957039,) * 3


def histogram(cls, values):
    return cls(
        mprime_edges=jnp.linspace(0, 1, 5),
        thetaprime_edges=jnp.linspace(0, 0.5, 4),
        values=jnp.asarray(values),
        mother_mass=MASS,
        masses=MASSES,
        folded=True,
    )


def test_interpolation_recovers_affine_map_with_folding_and_outer_plateaus():
    xc = (np.arange(4) + 0.5) / 4
    yc = (np.arange(3) + 0.5) / 6
    hist = histogram(LauraEfficiency, 1 + 2 * xc[:, None] + 3 * yc[None, :])
    mp = np.array([0.01, 0.20, 0.47, 0.82, 0.99])
    tp = np.array([0.01, 0.31, 0.49, 0.85, 0.99])
    invariants = square_dalitz_to_invariants(
        mp,
        tp,
        mother_mass=MASS,
        masses=MASSES,
    )
    data = dict(zip(("s12", "s13", "s23"), invariants, strict=True))
    expected = (
        1
        + 2 * np.clip(mp, xc[0], xc[-1])
        + 3 * np.clip(np.minimum(tp, 1 - tp), yc[0], yc[-1])
    )
    np.testing.assert_allclose(hist(data), expected, rtol=1e-10)


def test_uniform_sdp_density_integrates_to_unit_square_area():
    sample = SquareDalitzGrid(
        MASS, MASSES, resolution=40, quadrature="midpoint"
    ).sample()
    background = histogram(LauraBackground, np.ones((4, 3)))
    efficiency = histogram(LauraEfficiency, np.ones((4, 3)))
    # A constant SDP density has integral one on the full square. A constant
    # efficiency instead leaves the invariant-plane phase-space area intact.
    integral = jnp.mean(sample.weights * background(sample.as_dict()))
    np.testing.assert_allclose(integral, 1.0, rtol=1e-9)
    np.testing.assert_allclose(efficiency(sample.as_dict()), 1.0)
    assert float(jnp.mean(sample.weights)) > 100


def test_post_veto_scaling_preserves_counting_asymmetry_for_different_shapes():
    sample = SquareDalitzGrid(MASS, MASSES, resolution=40).sample()
    veto = MassWindowVeto((0, 2), 1.74, 1.894)
    asymmetry = -0.0078
    data = sample.as_dict()
    plus = histogram(LauraBackground, np.arange(1, 13).reshape(4, 3))
    minus = histogram(LauraBackground, np.arange(12, 0, -1).reshape(4, 3))
    pv = charge_scaled_background(plus, sample, veto, asymmetry, +1)(data) * veto(data)
    mv = charge_scaled_background(minus, sample, veto, asymmetry, -1)(data) * veto(data)
    category = CPBackgroundCategory(
        "qqbar",
        pv,
        mv,
        jnp.mean(sample.weights * pv),
        jnp.mean(sample.weights * mv),
    )
    np.testing.assert_allclose(category.plus_probability, 0.5039, atol=1e-12, rtol=0)
    np.testing.assert_allclose(category.minus_probability, 0.4961, atol=1e-12, rtol=0)
