import jax.numpy as jnp

from dalitzplotfitter.dynamics import LauraCovariantRBW
from dalitzplotfitter.kinematics import PhasespaceMC


FINAL_STATE = ("pi-", "pi+", "pi+")


def _data():
    sample = PhasespaceMC(
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
    ).generate(3, seed=9281)
    return sample.as_dict()


def _component(spin, daughter_key="p1", partner_key="p2", bachelor_key="p3", *, final_state=None):
    return LauraCovariantRBW(
        mass0=0.77526,
        width0=0.1491,
        parent_mass=1.86966,
        daughter_masses=(0.13957, 0.13957),
        bachelor_mass=0.13957,
        angular_momentum=spin,
        resonance_radius=1.5,
        parent_radius=5.0,
        daughter_key=daughter_key,
        partner_key=partner_key,
        bachelor_key=bachelor_key,
        final_state=final_state,
    )


def test_covariant_rbw_is_finite_for_physical_events():
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
