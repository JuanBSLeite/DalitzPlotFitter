import jax.numpy as jnp
import pytest

from dalitzplotfitter import Parameter, enable_x64
from dalitzplotfitter.dynamics import ResonanceAmplitude, ResonanceContext
from dalitzplotfitter.kinematics import PhaseSpaceMC


enable_x64()

FINAL_STATE = ("pi-", "pi+", "pi+")


def _data():
    sample = PhaseSpaceMC(
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
    ).generate(3, seed=9281)
    return sample.as_dict()


def _component(
    spin,
    daughter_key="p1",
    partner_key="p2",
    bachelor_key="p3",
    *,
    final_state=None,
):
    context = ResonanceContext(
        parent_mass=1.86966,
        daughter_masses=(0.13957, 0.13957),
        bachelor_mass=0.13957,
        spin=spin,
        pole_mass=0.77526,
        pole_width=0.1491,
        resonance_radius=1.5,
        parent_radius=5.0,
    )
    return ResonanceAmplitude(
        context=context,
        daughter_key=daughter_key,
        partner_key=partner_key,
        bachelor_key=bachelor_key,
        final_state=final_state,
    )


def _floating_mass_component(*, floating_width=False):
    mass = Parameter.dynamics(
        "rho.mass",
        0.77526,
        owner="rho",
        backend_name="mass",
        bounds=(0.70, 0.85),
    )
    width = (
        Parameter.dynamics(
            "rho.width",
            0.1491,
            owner="rho",
            backend_name="width",
            bounds=(0.08, 0.25),
        )
        if floating_width
        else 0.1491
    )
    return ResonanceAmplitude(
        context=ResonanceContext(
            parent_mass=1.86966,
            daughter_masses=(0.13957, 0.13957),
            bachelor_mass=0.13957,
            spin=1,
            pole_mass=mass,
            pole_width=width,
            resonance_radius=1.5,
            parent_radius=5.0,
        )
    )


def test_resonance_amplitude_is_finite_for_physical_events():
    values = _component(1)(_data())
    assert values.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(values.real)))
    assert bool(jnp.all(jnp.isfinite(values.imag)))


def test_spin_one_daughter_exchange_flips_complete_amplitude_sign():
    data = _data()
    first = _component(1, "p1", "p2")(data)
    second = _component(1, "p2", "p1")(data)
    assert jnp.allclose(first, -second, rtol=1e-6, atol=1e-7)


def test_spin_zero_daughter_exchange_leaves_complete_amplitude_unchanged():
    data = _data()
    first = _component(0, "p1", "p2")(data)
    second = _component(0, "p2", "p1")(data)
    assert jnp.allclose(first, second, rtol=1e-6, atol=1e-7)


def test_identical_final_state_is_symmetrized_automatically():
    data = _data()
    automatic = _component(1, final_state=FINAL_STATE)(data)
    pair12 = _component(1, "p1", "p2", "p3")(data)
    pair13 = _component(1, "p1", "p3", "p2")(data)
    assert jnp.allclose(automatic, pair12 + pair13, rtol=1e-6, atol=1e-7)


def test_nonidentical_final_state_does_not_add_extra_pairing():
    data = _data()
    labels = ("a", "b", "c")
    automatic = _component(1, final_state=labels)(data)
    nominal = _component(1)(data)
    assert jnp.allclose(automatic, nominal, rtol=1e-6, atol=1e-7)


def test_three_identical_bosons_add_three_unique_scalar_pairings():
    data = _data()
    labels = ("pi0", "pi0", "pi0")
    automatic = _component(0, final_state=labels)(data)
    pair12 = _component(0, "p1", "p2", "p3")(data)
    pair13 = _component(0, "p1", "p3", "p2")(data)
    pair23 = _component(0, "p2", "p3", "p1")(data)
    assert jnp.allclose(
        automatic,
        pair12 + pair13 + pair23,
        rtol=1e-6,
        atol=1e-7,
    )


def test_odd_spin_identical_resonance_daughters_are_rejected():
    data = _data()
    labels = ("pi0", "pi0", "K0")
    with pytest.raises(ValueError, match="odd-spin resonance"):
        _component(1, final_state=labels)(data)


def test_floating_mass_fixed_width_preparation_matches_direct_evaluation():
    component = _floating_mass_component(floating_width=False)
    data = _data()
    prepared = component.prepare_data(data)

    assert any(key.endswith("_angular_prepared") for key in prepared)
    assert any(key.endswith("_res_barrier_denominator") for key in prepared)
    assert any(key.endswith("_parent_barrier_denominator") for key in prepared)

    for mass in (0.735, 0.77526, 0.825):
        parameters = {"mass": mass}
        direct = component(data, parameters)
        cached = component(prepared, parameters)
        assert jnp.allclose(cached, direct, rtol=1e-12, atol=1e-12)


def test_mass_only_preparation_is_disabled_when_width_also_floats():
    component = _floating_mass_component(floating_width=True)
    prepared = component.prepare_data(_data())
    assert not any(key.endswith("_angular_prepared") for key in prepared)
    assert not any(key.endswith("_res_barrier_denominator") for key in prepared)
    assert not any(key.endswith("_parent_barrier_denominator") for key in prepared)
