import jax
import jax.numpy as jnp

from dalitzplotfitter.kinematics import PhaseSpaceSample
from dalitzplotfitter.sampling import weighted_resample


def test_weighted_resample_prefers_high_weight_candidates():
    sample = PhaseSpaceSample(
        s12=jnp.asarray([0.0, 1.0]),
        s13=jnp.asarray([0.0, 1.0]),
        s23=jnp.asarray([0.0, 1.0]),
        weights=jnp.ones(2),
    )
    selected = weighted_resample(
        jax.random.key(11),
        sample,
        jnp.asarray([1.0, 9.0]),
        20_000,
    )
    high_fraction = jnp.mean(selected.s12)
    assert jnp.abs(high_fraction - 0.9) < 0.02
    assert jnp.all(selected.weights == 1.0)


def test_weighted_resample_rejects_negative_weights():
    sample = PhaseSpaceSample(
        s12=jnp.asarray([0.0, 1.0]),
        s13=jnp.asarray([0.0, 1.0]),
        s23=jnp.asarray([0.0, 1.0]),
        weights=jnp.ones(2),
    )
    try:
        weighted_resample(
            jax.random.key(12),
            sample,
            jnp.asarray([1.0, -1.0]),
            10,
        )
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative weights should be rejected")
