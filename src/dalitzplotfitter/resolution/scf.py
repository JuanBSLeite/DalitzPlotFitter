"""Self-cross-feed migration maps for Dalitz-plot resolution effects.

The implementation follows the Laura++ ``LauScfMap`` convention: for every
true Dalitz bin there is a normalized probability distribution over
reconstructed bins. The SCF contribution is evaluated from the probability
mass in each true bin and converted back to a reconstructed density using the
reconstructed-bin phase-space area.

Migration may be stored densely or as a compact COO operator. The sparse path
uses only JAX gather/scatter operations and remains differentiable with respect
to the true density.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dalitzplotfitter.kinematics import (
    invariants_to_square_dalitz,
    square_dalitz_jacobian,
    square_dalitz_to_invariants,
)


@dataclass(frozen=True)
class SparseMigration:
    """COO representation of ``P(reco_bin | true_bin)``.

    Parameters are one-dimensional arrays of equal length. Multiple entries
    with the same ``(true_bin, reco_bin)`` pair are allowed and are summed by
    both the row-normalization check and migration operation.

    Construct this object directly for large SCF maps so a dense
    ``n_bins x n_bins`` matrix never has to exist in host or accelerator memory.
    """

    true_indices: Array
    reco_indices: Array
    probabilities: Array
    n_bins: int

    def __post_init__(self) -> None:
        true_indices = jnp.asarray(self.true_indices, dtype=jnp.int32)
        reco_indices = jnp.asarray(self.reco_indices, dtype=jnp.int32)
        probabilities = jnp.asarray(self.probabilities, dtype=jnp.float64)
        if self.n_bins < 1:
            raise ValueError("SparseMigration n_bins must be positive")
        if true_indices.ndim != 1 or reco_indices.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("SparseMigration arrays must be one-dimensional")
        if not (
            true_indices.shape == reco_indices.shape == probabilities.shape
        ):
            raise ValueError("SparseMigration arrays must have identical shapes")
        if bool(jnp.any(true_indices < 0)) or bool(jnp.any(true_indices >= self.n_bins)):
            raise ValueError("SparseMigration true-bin index is out of range")
        if bool(jnp.any(reco_indices < 0)) or bool(jnp.any(reco_indices >= self.n_bins)):
            raise ValueError("SparseMigration reco-bin index is out of range")
        if bool(jnp.any(~jnp.isfinite(probabilities))):
            raise ValueError("SparseMigration probabilities must be finite")
        if bool(jnp.any(probabilities < 0.0)):
            raise ValueError("migration probabilities must be non-negative")
        object.__setattr__(self, "true_indices", true_indices)
        object.__setattr__(self, "reco_indices", reco_indices)
        object.__setattr__(self, "probabilities", probabilities)

    @classmethod
    def from_dense(
        cls,
        migration: Array,
        *,
        zero_threshold: float = 0.0,
    ) -> "SparseMigration":
        """Compress a dense migration matrix once on the host."""

        if zero_threshold < 0.0:
            raise ValueError("zero_threshold must be non-negative")
        dense = np.asarray(jax.device_get(jnp.asarray(migration, dtype=jnp.float64)))
        if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
            raise ValueError("dense migration matrix must be square")
        mask = np.abs(dense) > float(zero_threshold)
        true_indices, reco_indices = np.nonzero(mask)
        probabilities = dense[true_indices, reco_indices]
        return cls(
            jnp.asarray(true_indices, dtype=jnp.int32),
            jnp.asarray(reco_indices, dtype=jnp.int32),
            jnp.asarray(probabilities, dtype=jnp.float64),
            int(dense.shape[0]),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_bins, self.n_bins

    @property
    def nnz(self) -> int:
        return int(self.probabilities.shape[0])

    @property
    def density(self) -> float:
        return self.nnz / float(self.n_bins * self.n_bins)

    def row_sums(self) -> Array:
        return jnp.zeros((self.n_bins,), dtype=self.probabilities.dtype).at[
            self.true_indices
        ].add(self.probabilities)

    def transpose_matvec(self, true_mass: Array) -> Array:
        """Return ``migration.T @ true_mass`` without materializing a matrix."""

        values = jnp.asarray(true_mass)
        if values.shape != (self.n_bins,):
            raise ValueError(
                f"true_mass must have shape ({self.n_bins},), got {values.shape}"
            )
        contributions = self.probabilities * values[self.true_indices]
        return jnp.zeros((self.n_bins,), dtype=contributions.dtype).at[
            self.reco_indices
        ].add(contributions)

    def to_dense(self) -> Array:
        """Materialize the migration matrix for diagnostics only."""

        return jnp.zeros((self.n_bins, self.n_bins), dtype=self.probabilities.dtype).at[
            self.true_indices, self.reco_indices
        ].add(self.probabilities)


@dataclass(frozen=True)
class SquareDalitzSCFMap:
    """Uniform Square-Dalitz SCF fraction and migration map.

    Fixed Square-Dalitz bin centres, invariant coordinates and phase-space
    areas are cached after their first construction. They depend only on the
    map geometry and therefore must not be rebuilt on every likelihood call.

    Parameters
    ----------
    migration:
        Either a dense array with shape ``(n_true_bins, n_reco_bins)`` or a
        :class:`SparseMigration`. Each non-empty true row is a conditional
        probability distribution ``P(reco_bin | true_bin)`` and must sum to one.
    scf_fraction:
        Fraction of reconstructed signal events that are SCF for each true bin.
        Values must lie in ``[0, 1]``.
    storage:
        ``"auto"`` compresses a dense migration matrix when its non-zero
        fraction is at most ``sparse_threshold``. ``"dense"`` always retains
        dense storage and ``"sparse"`` always compresses it. Passing an already
        prepared :class:`SparseMigration` always uses sparse storage.
    sparse_threshold:
        Maximum dense non-zero fraction for automatic COO compression.
    """

    migration: Array | SparseMigration
    scf_fraction: Array
    mother_mass: float
    masses: tuple[float, float, float]
    bins_mprime: int
    bins_thetaprime: int
    pair: tuple[int, int] = (0, 1)
    normalization_tolerance: float = 1e-8
    storage: Literal["auto", "dense", "sparse"] = "auto"
    sparse_threshold: float = 0.25
    sparse_zero_threshold: float = 0.0

    def __post_init__(self) -> None:
        fraction = jnp.asarray(self.scf_fraction, dtype=jnp.float64)
        n_bins = int(self.bins_mprime) * int(self.bins_thetaprime)
        if self.bins_mprime < 1 or self.bins_thetaprime < 1:
            raise ValueError("SCF bin counts must be positive")
        if self.storage not in {"auto", "dense", "sparse"}:
            raise ValueError("SCF storage must be 'auto', 'dense', or 'sparse'")
        if not 0.0 <= self.sparse_threshold <= 1.0:
            raise ValueError("sparse_threshold must lie in [0, 1]")
        if self.sparse_zero_threshold < 0.0:
            raise ValueError("sparse_zero_threshold must be non-negative")
        if fraction.shape != (n_bins,):
            raise ValueError(
                f"scf_fraction must have shape ({n_bins},), got {fraction.shape}"
            )
        if bool(jnp.any((fraction < 0.0) | (fraction > 1.0))):
            raise ValueError("scf_fraction values must lie in [0, 1]")

        migration = self.migration
        if isinstance(migration, SparseMigration):
            if migration.shape != (n_bins, n_bins):
                raise ValueError(
                    f"migration must have shape ({n_bins}, {n_bins}), got {migration.shape}"
                )
            operator: Array | SparseMigration = migration
        else:
            dense = jnp.asarray(migration, dtype=jnp.float64)
            if dense.shape != (n_bins, n_bins):
                raise ValueError(
                    f"migration must have shape ({n_bins}, {n_bins}), got {dense.shape}"
                )
            if bool(jnp.any(~jnp.isfinite(dense))):
                raise ValueError("migration probabilities must be finite")
            if bool(jnp.any(dense < 0.0)):
                raise ValueError("migration probabilities must be non-negative")

            use_sparse = self.storage == "sparse"
            if self.storage == "auto":
                nnz = int(
                    jax.device_get(
                        jnp.count_nonzero(
                            jnp.abs(dense) > self.sparse_zero_threshold
                        )
                    )
                )
                use_sparse = nnz <= self.sparse_threshold * n_bins * n_bins
            operator = (
                SparseMigration.from_dense(
                    dense,
                    zero_threshold=self.sparse_zero_threshold,
                )
                if use_sparse
                else dense
            )

        row_sums = (
            operator.row_sums()
            if isinstance(operator, SparseMigration)
            else jnp.sum(operator, axis=1)
        )
        nonempty = row_sums > 0.0
        if bool(
            jnp.any(
                nonempty
                & (jnp.abs(row_sums - 1.0) > self.normalization_tolerance)
            )
        ):
            raise ValueError("each non-empty true-bin migration row must sum to one")
        if bool(jnp.any((fraction > 0.0) & ~nonempty)):
            raise ValueError(
                "a true bin with non-zero SCF fraction requires a migration distribution"
            )

        object.__setattr__(self, "migration", operator)
        object.__setattr__(self, "scf_fraction", fraction)

    @property
    def n_bins(self) -> int:
        return self.bins_mprime * self.bins_thetaprime

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.migration, SparseMigration)

    @property
    def migration_nnz(self) -> int:
        if isinstance(self.migration, SparseMigration):
            return self.migration.nnz
        return int(
            jax.device_get(
                jnp.count_nonzero(jnp.abs(self.migration) > self.sparse_zero_threshold)
            )
        )

    @property
    def migration_density(self) -> float:
        return self.migration_nnz / float(self.n_bins * self.n_bins)

    def migration_matrix(self) -> Array:
        """Return a dense migration matrix for diagnostics or export."""

        if isinstance(self.migration, SparseMigration):
            return self.migration.to_dense()
        return self.migration

    @property
    def bin_widths(self) -> tuple[float, float]:
        return 1.0 / self.bins_mprime, 1.0 / self.bins_thetaprime

    @cached_property
    def _square_centers(self) -> tuple[Array, Array]:
        mp = (
            jnp.arange(self.bins_mprime, dtype=jnp.float64) + 0.5
        ) / self.bins_mprime
        tp = (
            jnp.arange(self.bins_thetaprime, dtype=jnp.float64) + 0.5
        ) / self.bins_thetaprime
        grid_mp, grid_tp = jnp.meshgrid(mp, tp, indexing="ij")
        return grid_mp.reshape(-1), grid_tp.reshape(-1)

    def square_centers(self) -> tuple[Array, Array]:
        return self._square_centers

    @cached_property
    def _true_bin_data(self) -> dict[str, Array]:
        mp, tp = self._square_centers
        s12, s13, s23 = square_dalitz_to_invariants(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        return {"s12": s12, "s13": s13, "s23": s23}

    def true_bin_data(self) -> dict[str, Array]:
        """Return cached invariant coordinates at all true-bin centres."""

        return self._true_bin_data

    @cached_property
    def _phase_space_areas(self) -> Array:
        mp, tp = self._square_centers
        jacobian = square_dalitz_jacobian(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        dmp, dtp = self.bin_widths
        return jacobian * dmp * dtp

    def phase_space_areas(self) -> Array:
        """Return cached ``Delta Omega`` for every uniform SDP bin."""

        return self._phase_space_areas

    def bin_indices(self, mprime: Array, thetaprime: Array) -> Array:
        mp = jnp.asarray(mprime)
        tp = jnp.asarray(thetaprime)
        if mp.shape != tp.shape:
            raise ValueError("mprime and thetaprime must have identical shapes")
        i = jnp.floor(mp * self.bins_mprime).astype(jnp.int32)
        j = jnp.floor(tp * self.bins_thetaprime).astype(jnp.int32)
        i = jnp.clip(i, 0, self.bins_mprime - 1)
        j = jnp.clip(j, 0, self.bins_thetaprime - 1)
        return i * self.bins_thetaprime + j

    def bin_indices_from_invariants(
        self,
        s12: Array,
        s13: Array,
        s23: Array,
    ) -> Array:
        mp, tp = invariants_to_square_dalitz(
            s12,
            s13,
            s23,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        return self.bin_indices(mp, tp)

    def scf_fraction_at(
        self,
        s12: Array,
        s13: Array,
        s23: Array,
    ) -> Array:
        indices = self.bin_indices_from_invariants(s12, s13, s23)
        return self.scf_fraction[indices]

    def smeared_bin_density(self, true_density: Array) -> Array:
        """Migrate the SCF part of a true density to reconstructed bins.

        ``true_density`` is a density with respect to ordinary Dalitz phase
        space evaluated at the true-bin centres. It is first converted to
        probability mass using ``Delta Omega_true``. The migration operator is
        then applied and the result divided by ``Delta Omega_reco``. This is the
        discrete Laura++ Eq. (31), including the Jacobian ratio.
        """

        density = jnp.asarray(true_density)
        if density.shape != (self.n_bins,):
            raise ValueError(
                f"true_density must have shape ({self.n_bins},), got {density.shape}"
            )
        areas = self._phase_space_areas
        true_scf_mass = self.scf_fraction * density * areas
        if isinstance(self.migration, SparseMigration):
            reco_scf_mass = self.migration.transpose_matvec(true_scf_mass)
        else:
            reco_scf_mass = self.migration.T @ true_scf_mass
        return jnp.where(areas > 0.0, reco_scf_mass / areas, 0.0)

    def smeared_density_at(
        self,
        true_density: Array,
        s12: Array,
        s13: Array,
        s23: Array,
    ) -> Array:
        reco_density = self.smeared_bin_density(true_density)
        indices = self.bin_indices_from_invariants(s12, s13, s23)
        return reco_density[indices]


__all__ = ["SparseMigration", "SquareDalitzSCFMap"]
