import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    NeutralMesonMixing,
    NonResonant,
    Parameter,
    RealImag,
    TimeDependentFitSession,
    generate_time_dependent_toy,
)


def _session(size=8, **kwargs):
    channel = DecayChannel("D0", ("K(S)0", "pi+", "pi-"))
    model = DecayModel(
        channel,
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=8,
    )
    data = model.generate_phase_space(size, seed=3, include_momenta=False)
    return TimeDependentFitSession(
        model,
        data,
        jnp.linspace(0.05, 1.0, size),
        jnp.asarray([1, -1] * (size // 2)),
        NeutralMesonMixing(0.01, 0.005, 0.4103),
        time_range=(0.0, 4.0),
        **kwargs,
    )


def test_time_dependent_toy_returns_joint_sample_and_tags():
    toy = generate_time_dependent_toy(
        _session(), 40, production_fraction=0.75, seed=7, proposal_size=800
    )
    assert toy.data.size == 40
    assert toy.times.shape == (40,)
    assert toy.tags.shape == (40,)
    assert toy.true_tags.shape == (40,)
    assert np.all(np.isin(np.asarray(toy.tags), (-1, 1)))
    assert np.all(np.asarray(toy.times) >= 0.0)


def test_production_fraction_derives_observed_tag_fraction_in_extended_mode():
    session = _session(
        extended=True,
        signal_yield=Parameter("ns", 10.0),
        production_fraction=0.8,
        wrong_tag=0.1,
    )
    values = {p.name: p.value for p in session.parameters if not p.fixed}
    expected = 0.8 * 0.9 + 0.2 * 0.1
    assert float(session._signal_tag_fraction.resolve(values)) == pytest.approx(
        expected
    )
    assert np.isfinite(float(session.objective(values)))