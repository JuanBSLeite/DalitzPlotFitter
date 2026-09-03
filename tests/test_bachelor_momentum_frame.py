import jax.numpy as jnp
import pytest

from dalitzplotfitter import DecayChannel, DecayModel, RealImag, Resonance, enable_x64
from dalitzplotfitter.dynamics import ResonanceAmplitude, ResonanceContext
from dalitzplotfitter.dynamics.lineshape import (
    bachelor_momentum_parent_frame,
    bachelor_momentum_resonance_frame,
)
from dalitzplotfitter.kinematics import PhaseSpaceMC


enable_x64()


def _context():
    return ResonanceContext(
        parent_mass=1.96835,
        daughter_masses=(0.13957, 0.13957),
        bachelor_mass=0.13957,
        spin=1,
        pole_mass=0.77526,
        pole_width=0.1491,
        resonance_radius=1.5,
        parent_radius=5.0,
    )


def _data():
    return PhaseSpaceMC(
        mother_mass=1.96835,
        masses=(0.13957, 0.13957, 0.13957),
    ).generate(8, seed=2026).as_dict()


def test_parent_and_resonance_frame_momenta_use_expected_denominators():
    context = _context()
    mass = jnp.asarray([0.6, 0.9, 1.2])
    p_res = bachelor_momentum_resonance_frame(
        context.parent_mass, mass, context.bachelor_mass
    )
    p_parent = bachelor_momentum_parent_frame(
        context.parent_mass, mass, context.bachelor_mass
    )
    assert jnp.allclose(
        p_parent,
        p_res * mass / context.parent_mass,
        rtol=1e-12,
        atol=1e-12,
    )


def test_resonance_frame_is_backward_compatible_default():
    context = _context()
    data = _data()
    default = ResonanceAmplitude(context=context)(data)
    explicit = ResonanceAmplitude(
        context=context,
        bachelor_momentum_frame="resonance",
    )(data)
    assert jnp.allclose(default, explicit, rtol=1e-12, atol=1e-12)


def test_parent_frame_changes_parent_barrier_factor_for_nonzero_spin():
    context = _context()
    data = _data()
    resonance_frame = ResonanceAmplitude(
        context=context,
        bachelor_momentum_frame="resonance",
    )(data)
    parent_frame = ResonanceAmplitude(
        context=context,
        bachelor_momentum_frame="parent",
    )(data)
    assert not bool(jnp.allclose(resonance_frame, parent_frame, rtol=1e-8, atol=1e-10))
    assert bool(jnp.all(jnp.isfinite(parent_frame.real)))
    assert bool(jnp.all(jnp.isfinite(parent_frame.imag)))


def test_high_level_resonance_propagates_bachelor_momentum_frame():
    channel = DecayChannel("D_s+", ("pi-", "pi+", "pi+"))
    component = Resonance(
        "rho(770)0",
        (0, 1),
        RealImag(1.0, 0.0),
        mass=0.77526,
        width=0.1491,
        spin=1,
        bachelor_momentum_frame="parent",
    )
    model = DecayModel(channel, (component,))
    dynamics = model.amplitude_model.components[0].function
    assert dynamics.bachelor_momentum_frame == "parent"


def test_invalid_bachelor_momentum_frame_is_rejected():
    context = _context()
    with pytest.raises(ValueError, match="bachelor_momentum_frame"):
        ResonanceAmplitude(context=context, bachelor_momentum_frame="lab")

    with pytest.raises(ValueError, match="bachelor_momentum_frame"):
        Resonance(
            "rho(770)0",
            (0, 1),
            RealImag(1.0, 0.0),
            mass=0.77526,
            width=0.1491,
            spin=1,
            bachelor_momentum_frame="lab",
        )
