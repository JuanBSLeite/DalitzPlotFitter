import jax.numpy as jnp

from dalitzplotfitter import KMatrix, RealImag, ResonanceContext, enable_x64


enable_x64()


def _context():
    mpi = 0.13957039
    return ResonanceContext(
        parent_mass=1.86965,
        daughter_masses=(mpi, mpi),
        bachelor_mass=mpi,
        spin=0,
        pole_mass=1.0,
        pole_width=0.0,
        resonance_radius=3.0,
        parent_radius=3.0,
    )


def test_kmatrix_scattering_matrix_is_symmetric():
    model = KMatrix()
    masses = jnp.asarray([0.35, 0.80, 1.00, 1.30, 1.70])
    matrix = model.scattering_matrix(masses)
    assert matrix.shape == (5, 5, 5)
    assert bool(jnp.allclose(matrix, jnp.swapaxes(matrix, -1, -2), rtol=0.0, atol=1e-12))


def test_kmatrix_phase_space_has_five_channels_and_analytic_continuation():
    model = KMatrix()
    rho = model.phase_space(jnp.asarray(0.80))
    assert rho.shape == (5,)
    assert abs(float(jnp.imag(rho[0]))) < 1e-12
    assert float(jnp.real(rho[0])) > 0.0
    assert abs(float(jnp.real(rho[1]))) < 1e-12
    assert float(jnp.imag(rho[1])) > 0.0


def test_kmatrix_pvector_and_amplitude_are_finite_away_from_bare_poles():
    model = KMatrix(
        betas=(
            RealImag(1.0, 0.0),
            RealImag(0.2, -0.1),
            0.0j,
            0.0j,
            0.0j,
        )
    )
    masses = jnp.asarray([0.31, 0.50, 0.90, 1.10, 1.40, 1.75])
    production = model.production_vector(masses)
    amplitude = model.amplitude_vector(masses)
    assert production.shape == (6, 5)
    assert amplitude.shape == (6, 5)
    assert bool(jnp.all(jnp.isfinite(jnp.real(amplitude))))
    assert bool(jnp.all(jnp.isfinite(jnp.imag(amplitude))))


def test_kmatrix_returns_pipi_channel_and_rejects_non_scalar_context():
    model = KMatrix()
    context = _context()
    mass = jnp.asarray(0.90)
    vector = model.amplitude_vector(mass)
    value = model(mass, context)
    assert abs(complex(value - vector[0])) < 1e-12

    bad = ResonanceContext(
        parent_mass=context.parent_mass,
        daughter_masses=context.daughter_masses,
        bachelor_mass=context.bachelor_mass,
        spin=1,
        pole_mass=context.pole_mass,
        pole_width=context.pole_width,
    )
    try:
        model(mass, bad)
    except ValueError as exc:
        assert "scalar" in str(exc)
    else:
        raise AssertionError("KMatrix accepted a non-scalar context")


def test_kmatrix_production_coefficients_change_amplitude_linearly_before_rescattering():
    masses = jnp.asarray([0.50, 0.90, 1.30])
    first = KMatrix(betas=(1.0j, 0.0j, 0.0j, 0.0j, 0.0j))
    doubled = KMatrix(betas=(2.0j, 0.0j, 0.0j, 0.0j, 0.0j))
    a1 = first.amplitude_vector(masses)
    a2 = doubled.amplitude_vector(masses)
    assert bool(jnp.allclose(a2, 2.0 * a1, rtol=1e-11, atol=1e-11))
