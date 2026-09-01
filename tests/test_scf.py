import jax.numpy as jnp

from dalitzplotfitter import SCFSignalPDF, SquareDalitzGrid, SquareDalitzSCFMap, enable_x64
from dalitzplotfitter.integration import GridIntegrator


enable_x64()


def _map(migration, fraction):
    return SquareDalitzSCFMap(
        migration=jnp.asarray(migration, dtype=jnp.float64),
        scf_fraction=jnp.asarray(fraction, dtype=jnp.float64),
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
        bins_mprime=2,
        bins_thetaprime=2,
        pair=(0, 1),
    )


def test_scf_migration_conserves_scf_probability_mass():
    migration = jnp.asarray(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.75],
            [0.4, 0.0, 0.0, 0.6],
        ]
    )
    fraction = jnp.asarray([0.2, 0.4, 0.6, 0.8])
    scf = _map(migration, fraction)
    density = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    areas = scf.phase_space_areas()

    source_mass = jnp.sum(fraction * density * areas)
    migrated_density = scf.smeared_bin_density(density)
    migrated_mass = jnp.sum(migrated_density * areas)

    assert jnp.allclose(migrated_mass, source_mass, rtol=1e-12, atol=1e-12)


def test_identity_migration_recovers_true_density_at_bin_centres():
    scf = _map(jnp.eye(4), jnp.asarray([0.1, 0.3, 0.5, 0.7]))
    density = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    smeared = scf.smeared_bin_density(density)
    expected = scf.scf_fraction * density
    assert jnp.allclose(smeared, expected, rtol=1e-12, atol=1e-12)


def test_scf_signal_pdf_reduces_to_unsmeared_signal_for_identity_migration():
    grid = SquareDalitzGrid(
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
        resolution=2,
        pair=(0, 1),
        quadrature="midpoint",
    ).sample()
    scf = _map(jnp.eye(4), jnp.asarray([0.2, 0.4, 0.6, 0.8]))
    integrator = GridIntegrator(grid)

    def intensity(data, parameters):
        return 1.0 + 0.2 * data["s12"] + 0.1 * data["s13"]

    pdf = SCFSignalPDF(intensity=intensity, integrator=integrator, scf_map=scf)
    data = scf.true_bin_data()
    base = intensity(data, {})
    expected = base / integrator.integrate(lambda d: intensity(d, {}))

    assert jnp.allclose(pdf(data, {}), expected, rtol=1e-12, atol=1e-12)


def test_scf_map_rejects_non_normalized_true_rows():
    bad = jnp.eye(4).at[0, 0].set(0.7)
    try:
        _map(bad, jnp.ones(4) * 0.2)
    except ValueError as exc:
        assert "sum to one" in str(exc)
    else:
        raise AssertionError("non-normalized migration row should fail")
