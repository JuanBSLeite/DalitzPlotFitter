import jax.numpy as jnp

from dalitzplotfitter import BaBarFlatte, ResonanceContext


def _context():
    return ResonanceContext(
        parent_mass=5.279,
        daughter_masses=(0.13957039, 0.13957039),
        bachelor_mass=0.493677,
        spin=0,
        pole_mass=0.965,
        pole_width=0.0,
        resonance_radius=4.0,
        parent_radius=4.0,
    )


def test_babar_flatte_uses_isospin_weighted_phase_space():
    model = BaBarFlatte()
    mass = jnp.asarray(1.10)
    gamma_pi, gamma_k = model.widths(mass)

    rho = lambda mh: jnp.sqrt((1.0 - 4.0 * mh**2 / mass**2).astype(jnp.complex128))
    expected_pi = model.g_pi * ((1.0 / 3.0) * rho(model.mpi0) + (2.0 / 3.0) * rho(model.mpip))
    expected_k = model.g_k * (0.5 * rho(model.mkp) + 0.5 * rho(model.mk0))

    assert jnp.allclose(gamma_pi, expected_pi)
    assert jnp.allclose(gamma_k, expected_k)


def test_babar_flatte_analytically_continues_below_kk_threshold():
    model = BaBarFlatte()
    _, gamma_k = model.widths(jnp.asarray(0.965))
    assert jnp.imag(gamma_k) > 0.0
    value = model(jnp.asarray(0.965), _context())
    assert jnp.isfinite(jnp.real(value))
    assert jnp.isfinite(jnp.imag(value))
