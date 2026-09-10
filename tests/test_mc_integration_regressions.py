from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    FunctionalVeto,
    NonResonant,
    RealImag,
)
from dalitzplotfitter.kinematics import PhaseSpaceMC, PhaseSpaceSample


def sample():
    return PhaseSpaceSample(
        jnp.array([1.0, 2.0, 3.0, 4.0]),
        jnp.ones(4),
        jnp.ones(4),
        jnp.array([0.5, 1.0, 2.0, 3.0]),
    )


def model(**kwargs):
    return DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=12,
        **kwargs,
    )


@pytest.mark.parametrize("field", ["s12", "s13", "s23", "weights"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_nonfinite_external_samples(field, bad):
    s = sample()
    s = replace(s, **{field: getattr(s, field).at[0].set(bad)})
    with pytest.raises(ValueError, match="finite"):
        model(normalization_sample=s)
    m = model()
    with pytest.raises(ValueError, match="finite"):
        m.pdf(s)
    with pytest.raises(ValueError, match="finite"):
        m.prepare_cache(sample(), s)


@pytest.mark.parametrize("weights", [jnp.zeros(4), -jnp.ones(4)])
def test_rejects_unusable_weights(weights):
    with pytest.raises(ValueError, match="positive"):
        model(normalization_sample=replace(sample(), weights=weights))


def test_selection_preserves_integral_and_data_selection():
    s = sample()
    veto = FunctionalVeto(lambda d: d["s12"] < 3)
    selected = veto.apply(s, for_integration=True)
    for f in (lambda d: jnp.ones_like(d["s12"]), lambda d: d["s12"] ** 2):
        expected = jnp.mean(s.weights * f(s.as_dict()) * veto(s.as_dict()))
        assert np.isclose(jnp.mean(selected.weights * f(selected.as_dict())), expected)
    np.testing.assert_array_equal(veto.apply(s).weights, s.weights[:2])
    with pytest.raises(ValueError, match="no events"):
        s.select_for_integration(jnp.zeros(4, dtype=bool))


def test_importance_weights_replace_old_weights():
    s = sample()
    q = jnp.array([0.1, 0.2, 0.3, 0.4])
    result = s.with_importance_weights(q)
    np.testing.assert_allclose(result.weights, 1 / np.asarray(q))
    with pytest.raises(ValueError, match="proposal_density"):
        s.with_importance_weights(q.at[0].set(0.0))


def test_stratified_area_and_proposal_contract():
    mc = PhaseSpaceMC(2.0, (0.0, 0.0, 0.0))
    options = dict(cell_probabilities=jnp.array([9.0, 1.0]), grid_shape=(2, 1), seed=12)
    s, cells = mc.generate_stratified_invariants(100_000, **options)
    proposal, proposal_cells = mc.generate_stratified_invariants(
        100_000, integration_weights=False, **options
    )
    np.testing.assert_array_equal(cells, proposal_cells)
    np.testing.assert_allclose(
        s.weights, proposal.weights / (2 * jnp.array([0.9, 0.1])[cells])
    )
    values = np.asarray(s.weights) * (128 * np.pi**3 * 4)
    assert abs(values.mean() - 8.0) < 5 * values.std(ddof=1) / np.sqrt(s.size)
    assert abs(float(jnp.mean(proposal.weights)) * (128 * np.pi**3 * 4) - 8.0) > 2.0


@pytest.mark.parametrize("p", [[0.0, 1.0], [-1.0, 2.0], [np.nan, 1.0], [0.0, 0.0]])
def test_invalid_stratified_integration_probabilities(p):
    with pytest.raises(ValueError, match="probabilit"):
        PhaseSpaceMC(2.0, (0.0, 0.0, 0.0)).generate_stratified_invariants(
            10, cell_probabilities=jnp.array(p), grid_shape=(2, 1), seed=1
        )


def test_importance_reweighting_recovers_known_integral():
    # Inverse CDF points for q(x)=2*x on [0,1]. Unweighted mean(x) is 2/3,
    # whereas the target Lebesgue integral of x is 1/2.
    x = jnp.sqrt((jnp.arange(10000) + 0.5) / 10000)
    events = PhaseSpaceSample(x, jnp.ones_like(x), jnp.ones_like(x), jnp.ones_like(x))
    corrected = events.with_importance_weights(2 * x)
    assert abs(float(jnp.mean(x)) - 2 / 3) < 1e-4
    assert np.isclose(jnp.mean(corrected.weights * x), 0.5)
    assert abs(float(jnp.mean(corrected.weights)) - 1.0) < 0.01
