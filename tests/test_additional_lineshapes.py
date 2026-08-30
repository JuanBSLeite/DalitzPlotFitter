import jax.numpy as jnp

from dalitzplotfitter import Flatte, GounarisSakurai, ResonanceContext, enable_x64


enable_x64()


def _rho_context():
    mpi = 0.13957039
    return ResonanceContext(
        parent_mass=1.86965,
        daughter_masses=(mpi, mpi),
        bachelor_mass=mpi,
        spin=1,
        pole_mass=0.77526,
        pole_width=0.1491,
        resonance_radius=3.0,
        parent_radius=3.0,
    )


def _scalar_context(m0=0.965):
    mpi = 0.13957039
    return ResonanceContext(
        parent_mass=1.86965,
        daughter_masses=(mpi, mpi),
        bachelor_mass=mpi,
        spin=0,
        pole_mass=m0,
        pole_width=0.0,
        resonance_radius=3.0,
        parent_radius=3.0,
    )


def test_gounaris_sakurai_is_finite_and_purely_imaginary_at_pole():
    context = _rho_context()
    value = GounarisSakurai()(jnp.asarray(context.pole_mass), context)
    assert bool(jnp.isfinite(jnp.real(value)))
    assert bool(jnp.isfinite(jnp.imag(value)))
    assert abs(float(jnp.real(value))) < 1e-11
    assert float(jnp.imag(value)) > 0.0


def test_gounaris_sakurai_rejects_non_vector_context():
    context = _scalar_context()
    try:
        GounarisSakurai()(jnp.asarray(0.8), context)
    except ValueError as exc:
        assert "spin-1" in str(exc)
    else:
        raise AssertionError("GounarisSakurai accepted a non-vector context")


def test_f0_flatte_has_analytic_kaon_threshold_continuation():
    context = _scalar_context()
    flatte = Flatte.f0_980()

    # Between pi-pi and K-K thresholds the pion width is real while the kaon
    # width is purely imaginary, producing the characteristic dispersive cusp.
    gamma_pi, gamma_k = flatte.widths(jnp.asarray(0.97), context)
    assert abs(float(jnp.imag(gamma_pi))) < 1e-12
    assert float(jnp.real(gamma_pi)) > 0.0
    assert abs(float(jnp.real(gamma_k))) < 1e-12
    assert float(jnp.imag(gamma_k)) > 0.0

    value = flatte(jnp.asarray(0.97), context)
    assert bool(jnp.isfinite(jnp.real(value)))
    assert bool(jnp.isfinite(jnp.imag(value)))


def test_flatte_adler_zero_suppresses_both_channel_widths():
    context = _scalar_context(m0=1.43)
    flatte = Flatte.k0star_1430_neutral()
    mass = jnp.sqrt(jnp.asarray(flatte.adler_zero))
    gamma1, gamma2 = flatte.widths(mass, context)
    assert abs(complex(gamma1)) < 1e-12
    assert abs(complex(gamma2)) < 1e-12


def test_flatte_presets_match_laura_table_coupling_ratios():
    f0 = Flatte.f0_980()
    a0 = Flatte.a0_980_neutral()
    assert abs(float(f0.g2 / f0.g1) - 4.21) < 1e-12
    assert abs(float(a0.g2 / a0.g1) - 1.03) < 1e-12
