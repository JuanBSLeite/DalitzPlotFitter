from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    Resonance,
    SquareDalitzGrid,
    model_from_spec,
    model_to_spec,
)
from dalitzplotfitter.dynamics import blatt_weisskopf_from_momenta


def polynomial(z, spin):
    return (
        1,
        1 + z,
        9 + 3*z + z*z,
        225 + 45*z + 6*z*z + z**3,
        11025 + 1575*z + 135*z*z + 10*z**3 + z**4,
    )[spin]


@pytest.mark.parametrize("spin", range(5))
def test_raw_factor_matches_laura_primed_polynomial(spin):
    q = jnp.array([0., .1, .7, 1.2])
    raw = blatt_weisskopf_from_momenta(q, .7, spin, 4., normalize_at_pole=False)
    np.testing.assert_allclose(raw, 1/np.sqrt(polynomial(np.asarray(q*4)**2, spin)))
    assert float(blatt_weisskopf_from_momenta(.7, .7, spin, 4.)) == 1.


@pytest.mark.parametrize("spin", range(5))
def test_public_option_scales_complete_amplitude_and_survives_json(spin):
    channel = DecayChannel("B+", ("pi+", "pi+", "pi-"))
    component = Resonance(
        "test", (2, 0), 1., mass=1.2, width=.1, spin=spin,
        resonance_radius=4., parent_radius=4., normalize_component=False,
    )
    normalized = DecayModel(channel, [component], normalization_resolution=20)
    raw = DecayModel(channel, [replace(component, normalize_form_factors=False)],
                     normalization_resolution=20)
    restored = model_from_spec(model_to_spec(raw))
    assert restored.components[0].normalize_form_factors is False
    sample = SquareDalitzGrid(channel.parent_mass, channel.daughter_masses,
                             resolution=12).sample()
    data = sample.as_dict()
    pion = channel.daughter_masses[0]
    q0sq = 1.2**2/4-pion**2
    p0sq = ((channel.parent_mass**2-1.2**2-pion**2)/(2*1.2))**2-pion**2
    scale = np.sqrt(polynomial(q0sq*16, spin)*polynomial(p0sq*16, spin))
    expected = np.asarray(normalized.amplitude(data))/scale
    np.testing.assert_allclose(jax.jit(restored.amplitude)(data), expected,
                               rtol=2e-12, atol=1e-12)
    # Prepared evaluation must honor the same option, including symmetrization.
    function = restored.amplitude_model.components[0].function
    np.testing.assert_allclose(function(function.prepare_data(data)), expected,
                               rtol=2e-12, atol=1e-12)
