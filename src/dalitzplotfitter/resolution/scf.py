"""Self-cross-feed migration maps for Dalitz-plot resolution effects.

The implementation follows the Laura++ ``LauScfMap`` convention: for every
true Dalitz bin there is a normalized probability distribution over
reconstructed bins.  The SCF contribution is evaluated from the probability
mass in each true bin and converted back to a reconstructed density using the
reconstructed-bin phase-space area.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.kinematics import (
    invariants_to_square_dalitz,
    square_dalitz_jacobian,
    square_dalitz_to_invariants,
)


@dataclass(frozen=True)
class SquareDalitzSCFMap:
    """Uniform Square-Dalitz SCF fraction and migration map.

    Parameters
    ----------
    migration:
        Array with shape ``(n_true_bins, n_reco_bins)``.  Each non-empty true
        row is a conditional probability distribution
        ``P(reco_bin | true_bin)`` and must sum to one.
    scf_fraction:
        Fraction of reconstructed signal events that are SCF for each true
        bin. Values must lie in ``[0, 1]``.
    mother_mass, masses, pair:
        Kinematic definition used by the package Square-Dalitz transform.
    bins_mprime, bins_thetaprime:
        Number of uniform bins in the two Square-Dalitz directions.  Laura++
        requires the SCF fraction and every migration histogram to use the
        same binning; this class enforces that convention.
    """

    migration: Array
    scf_fraction: Array
    mother_mass: float
    masses: tuple[float, float, float]
    bins_mprime: int
    bins_thetaprime: int
    pair: tuple[int, int] = (0, 1)
    normalization_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        migration = jnp.asarray(self.migration, dtype=jnp.float64)
        fraction = jnp.asarray(self.scf_fraction, dtype=jnp.float64)
        n_bins = int(self.bins_mprime) * int(self.bins_thetaprime)
        if self.bins_mprime < 1 or self.bins_thetaprime < 1:
            raise ValueError("SCF bin counts must be positive")
        if migration.shape != (n_bins, n_bins):
            raise ValueError(
                f"migration must have shape ({n_bins}, {n_bins}), got {migration.shape}"
            )
        if fraction.shape != (n_bins,):
            raise ValueError(
                f"scf_fraction must have shape ({n_bins},), got {fraction.shape}"
            )
        if bool(jnp.any(migration < 0.0)):
            raise ValueError("migration probabilities must be non-negative")
        if bool(jnp.any((fraction < 0.0) | (fraction > 1.0))):
            raise ValueError("scf_fraction values must lie in [0, 1]")

        row_sums = jnp.sum(migration, axis=1)
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

        object.__setattr__(self, "migration", migration)
        object.__setattr__(self, "scf_fraction", fraction)

    @property
    def n_bins(self) -> int:
        return self.bins_mprime * self.bins_thetaprime

    @property
    def bin_widths(self) -> tuple[float, float]:
        return 1.0 / self.bins_mprime, 1.0 / self.bins_thetaprime

    def square_centers(self) -> tuple[Array, Array]:
        mp = (jnp.arange(self.bins_mprime, dtype=jnp.float64) + 0.5) / self.bins_mprime
        tp = (jnp.arange(self.bins_thetaprime, dtype=jnp.float64) + 0.5) / self.bins_thetaprime
        grid_mp, grid_tp = jnp.meshgrid(mp, tp, indexing="ij")
        return grid_mp.reshape(-1), grid_tp.reshape(-1)

    def true_bin_data(self) -> dict[str, Array]:
        """Return invariant coordinates at the centres of all true bins."""

        mp, tp = self.square_centers()
        s12, s13, s23 = square_dalitz_to_invariants(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        return {"s12": s12, "s13": s13, "s23": s23}

    def phase_space_areas(self) -> Array:
        """Approximate ``Delta Omega`` of every uniform SDP bin.

        For uniform ``m'``/``theta'`` binning this is
        ``Delta m' Delta theta' |J|``, exactly the factor entering Laura++'s
        discretized SCF expression.
        """

        mp, tp = self.square_centers()
        jacobian = square_dalitz_jacobian(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        dmp, dtp = self.bin_widths
        return jacobian * dmp * dtp

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
        self, s12: Array, s13: Array, s23: Array
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
        self, s12: Array, s13: Array, s23: Array
    ) -> Array:
        indices = self.bin_indices_from_invariants(s12, s13, s23)
        return self.scf_fraction[indices]

    def smeared_bin_density(self, true_density: Array) -> Array:
        """Migrate the SCF part of a true density to reconstructed bins.

        ``true_density`` is a density with respect to ordinary Dalitz phase
        space evaluated at the true-bin centres.  It is first converted to
        probability mass using ``Delta Omega_true``.  The migration matrix is
        then applied and the result divided by ``Delta Omega_reco``.  This is
        the discrete Laura++ Eq. (31), including the Jacobian ratio.
        """

        density = jnp.asarray(true_density)
        if density.shape != (self.n_bins,):
            raise ValueError(
                f"true_density must have shape ({self.n_bins},), got {density.shape}"
            )
        areas = self.phase_space_areas()
        true_scf_mass = self.scf_fraction * density * areas
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


__all__ = ["SquareDalitzSCFMap"]
