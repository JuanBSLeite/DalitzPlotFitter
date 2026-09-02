import jax
import jax.numpy as jnp

from dalitzplotfitter import (
    FunctionalVeto,
    SCFSignalPDF,
    SparseMigration,
    SquareDalitzGrid,
    SquareDalitzSCFMap,
    enable_x64,
)
from dalitzplotfitter.integration import GridIntegrator


enable_x64()


def _map(migration, fraction, **kwargs):
    return SquareDalitzSCFMap(
        migration=migration,
        scf_fraction=jnp.asarray(fraction, dtype=jnp.float64),
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
        bins_mprime=2,
        bins_thetaprime=2,
        pair=(0, 1),
        **kwargs,
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


def test_auto_storage_compresses_sparse_migration():
    scf = _map(
        jnp.eye(4),
        jnp.ones(4) * 0.2,
        storage="auto",
        sparse_threshold=0.30,
    )
    assert scf.is_sparse
    assert scf.migration_nnz == 4
    assert abs(scf.migration_density - 0.25) < 1e-12
    assert jnp.allclose(scf.migration_matrix(), jnp.eye(4), atol=0.0, rtol=0.0)


def test_dense_and_sparse_scf_paths_are_numerically_identical():
    migration = jnp.asarray(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.75],
            [0.4, 0.0, 0.0, 0.6],
        ],
        dtype=jnp.float64,
    )
    fraction = jnp.asarray([0.2, 0.4, 0.6, 0.8])
    dense = _map(migration, fraction, storage="dense")
    sparse = _map(migration, fraction, storage="sparse")
    density = jnp.asarray([1.2, 0.7, 2.3, 4.5])
    assert not dense.is_sparse
    assert sparse.is_sparse
    assert jnp.allclose(
        sparse.smeared_bin_density(density),
        dense.smeared_bin_density(density),
        rtol=1e-12,
        atol=1e-12,
    )


def test_sparse_migration_can_be_constructed_without_dense_matrix():
    migration = SparseMigration(
        true_indices=jnp.asarray([0, 0, 1, 2, 2, 3, 3]),
        reco_indices=jnp.asarray([0, 1, 1, 2, 3, 0, 3]),
        probabilities=jnp.asarray([0.5, 0.5, 1.0, 0.25, 0.75, 0.4, 0.6]),
        n_bins=4,
    )
    scf = _map(migration, jnp.asarray([0.2, 0.4, 0.6, 0.8]))
    assert scf.is_sparse
    assert scf.migration_nnz == 7


def test_sparse_migration_matvec_is_jittable_and_differentiable():
    migration = SparseMigration(
        true_indices=jnp.asarray([0, 0, 1, 2, 2, 3, 3]),
        reco_indices=jnp.asarray([0, 1, 1, 2, 3, 0, 3]),
        probabilities=jnp.asarray([0.5, 0.5, 1.0, 0.25, 0.75, 0.4, 0.6]),
        n_bins=4,
    )
    scf = _map(migration, jnp.ones(4) * 0.5)
    density = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    compiled = jax.jit(scf.smeared_bin_density)
    value = compiled(density)
    gradient = jax.grad(lambda x: jnp.sum(compiled(x)))(density)
    assert bool(jnp.all(jnp.isfinite(value)))
    assert bool(jnp.all(jnp.isfinite(gradient)))


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


def test_scf_veto_is_applied_in_reconstructed_space_and_renormalized():
    grid = SquareDalitzGrid(
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
        resolution=2,
        pair=(0, 1),
        quadrature="midpoint",
    ).sample()
    migration = jnp.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    scf = _map(migration, jnp.ones(4) * 0.5)

    veto = FunctionalVeto(lambda data: data["s13"] < jnp.median(grid.s13))

    def intensity(data, parameters):
        return jnp.ones_like(data["s12"])

    pdf = SCFSignalPDF(
        intensity=intensity,
        integrator=GridIntegrator(grid),
        scf_map=scf,
        veto=veto,
    )
    data = scf.true_bin_data()
    values = pdf(data, {})
    accepted = veto(data)

    assert jnp.all(values[~accepted] == 0.0)
    assert pdf.normalization({}) > 0.0


def test_scf_map_rejects_non_normalized_true_rows():
    bad = jnp.eye(4).at[0, 0].set(0.7)
    try:
        _map(bad, jnp.ones(4) * 0.2)
    except ValueError as exc:
        assert "sum to one" in str(exc)
    else:
        raise AssertionError("non-normalized migration row should fail")
