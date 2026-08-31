"""Adaptive Square-Dalitz integration driven by bilinear amplitude convergence."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.kinematics.sample import PhaseSpaceSample
from dalitzplotfitter.kinematics.square_dalitz import (
    square_dalitz_jacobian,
    square_dalitz_to_invariants,
)


@dataclass(frozen=True)
class AdaptiveSquareDalitzResult:
    """Adaptive normalization sample and leaf-cell diagnostics."""

    sample: PhaseSpaceSample
    leaf_bounds: np.ndarray
    leaf_depths: np.ndarray
    leaf_errors: np.ndarray
    mprime: np.ndarray
    thetaprime: np.ndarray

    @property
    def size(self) -> int:
        return self.sample.size

    @property
    def n_leaves(self) -> int:
        return int(self.leaf_bounds.shape[0])


@dataclass(frozen=True)
class AdaptiveSquareDalitzGrid:
    """Adaptive integration on the Square-Dalitz plane.

    Refinement is driven by convergence of the full raw component bilinear
    matrix ``J F_i^* F_j``. For every cell, a midpoint estimate is compared
    with the sum of four quarter-cell midpoint estimates. The cell is split if
    any numerically relevant matrix element exceeds ``tolerance``.

    The algorithm does not inspect resonance metadata such as mass or width.
    It therefore also works for LASS, Flatte, K-matrix, QMI and arbitrary
    direct Dalitz amplitudes.
    """

    mother_mass: float
    masses: tuple[float, float, float]
    pair: tuple[int, int] = (0, 1)
    base_resolution: int = 24
    min_depth: int = 1
    max_depth: int = 5
    tolerance: float = 0.02
    matrix_floor: float = 1e-8
    max_cells: int = 200_000

    def __post_init__(self) -> None:
        if self.base_resolution < 2:
            raise ValueError("base_resolution must be at least 2")
        if self.min_depth < 0:
            raise ValueError("min_depth must be non-negative")
        if self.max_depth < self.min_depth:
            raise ValueError("max_depth must be >= min_depth")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if not (0.0 < self.matrix_floor < 1.0):
            raise ValueError("matrix_floor must lie between 0 and 1")
        if self.max_cells < self.base_resolution**2:
            raise ValueError("max_cells is smaller than the base grid")
        i, j = self.pair
        if i == j or i not in (0, 1, 2) or j not in (0, 1, 2):
            raise ValueError("pair must contain two distinct indices from 0, 1, 2")

    def _sample_at(self, mprime: np.ndarray, thetaprime: np.ndarray) -> PhaseSpaceSample:
        mp = jnp.asarray(mprime)
        tp = jnp.asarray(thetaprime)
        s12, s13, s23 = square_dalitz_to_invariants(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        return PhaseSpaceSample(
            s12=s12,
            s13=s13,
            s23=s23,
            weights=jnp.ones_like(s12),
        )

    @staticmethod
    def _raw_components(model, sample: PhaseSpaceSample, values=None) -> np.ndarray:
        data = sample.as_dict()
        rows = []
        for component in model.amplitude_model.components:
            rows.append(np.asarray(component.function(data, values), dtype=np.complex128))
        if not rows:
            raise ValueError("adaptive integration requires at least one amplitude component")
        return np.stack(rows, axis=1)

    def _cell_errors(self, model, cells: np.ndarray, values=None) -> np.ndarray:
        # cells columns: x0, x1, y0, y1, depth
        x0, x1, y0, y1 = (cells[:, i] for i in range(4))
        dx = x1 - x0
        dy = y1 - y0
        area = dx * dy
        xc = 0.5 * (x0 + x1)
        yc = 0.5 * (y0 + y1)

        center = self._sample_at(xc, yc)
        jc = np.asarray(
            square_dalitz_jacobian(
                xc,
                yc,
                mother_mass=self.mother_mass,
                masses=self.masses,
                pair=self.pair,
            )
        )
        fc = self._raw_components(model, center, values)
        bc = jc[:, None, None] * np.conj(fc)[:, :, None] * fc[:, None, :]
        coarse = area[:, None, None] * bc

        offsets = np.asarray(
            [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)]
        )
        xf = (xc[:, None] + dx[:, None] * offsets[None, :, 0]).reshape(-1)
        yf = (yc[:, None] + dy[:, None] * offsets[None, :, 1]).reshape(-1)
        fine_sample = self._sample_at(xf, yf)
        jf = np.asarray(
            square_dalitz_jacobian(
                xf,
                yf,
                mother_mass=self.mother_mass,
                masses=self.masses,
                pair=self.pair,
            )
        )
        ff = self._raw_components(model, fine_sample, values)
        bf = jf[:, None, None] * np.conj(ff)[:, :, None] * ff[:, None, :]
        n_cells = cells.shape[0]
        bf = bf.reshape(n_cells, 4, bf.shape[1], bf.shape[2])
        fine = area[:, None, None] * np.mean(bf, axis=1)

        magnitude = np.maximum(np.abs(fine), np.abs(coarse))
        local_scale = np.max(magnitude, axis=(1, 2), keepdims=True)
        relevant = magnitude >= self.matrix_floor * np.maximum(local_scale, 1e-300)
        denominator = np.maximum(magnitude, self.matrix_floor * np.maximum(local_scale, 1e-300))
        relative = np.where(relevant, np.abs(fine - coarse) / denominator, 0.0)
        return np.max(relative, axis=(1, 2))

    @staticmethod
    def _split(cells: np.ndarray) -> np.ndarray:
        children = []
        for x0, x1, y0, y1, depth in cells:
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)
            next_depth = depth + 1
            children.extend(
                [
                    (x0, xm, y0, ym, next_depth),
                    (xm, x1, y0, ym, next_depth),
                    (x0, xm, ym, y1, next_depth),
                    (xm, x1, ym, y1, next_depth),
                ]
            )
        return np.asarray(children, dtype=np.float64)

    def build(self, model, values=None) -> AdaptiveSquareDalitzResult:
        """Build the adaptive grid using the model's raw amplitude components."""
        n = self.base_resolution
        edges = np.linspace(0.0, 1.0, n + 1)
        initial = []
        for ix in range(n):
            for iy in range(n):
                initial.append((edges[ix], edges[ix + 1], edges[iy], edges[iy + 1], 0.0))
        active = np.asarray(initial, dtype=np.float64)
        leaves: list[np.ndarray] = []
        leaf_errors: list[np.ndarray] = []

        while active.size:
            if active.shape[0] + sum(item.shape[0] for item in leaves) > self.max_cells:
                raise RuntimeError("adaptive Square-Dalitz grid exceeded max_cells")
            errors = self._cell_errors(model, active, values)
            depth = active[:, 4].astype(int)
            refine = ((depth < self.min_depth) | (errors > self.tolerance)) & (depth < self.max_depth)
            keep = ~refine
            if np.any(keep):
                leaves.append(active[keep])
                leaf_errors.append(errors[keep])
            active = self._split(active[refine]) if np.any(refine) else np.empty((0, 5))

        leaf_cells = np.concatenate(leaves, axis=0)
        errors = np.concatenate(leaf_errors, axis=0)
        x0, x1, y0, y1 = (leaf_cells[:, i] for i in range(4))
        dx = x1 - x0
        dy = y1 - y0
        area = dx * dy
        xc = 0.5 * (x0 + x1)
        yc = 0.5 * (y0 + y1)
        offsets = np.asarray(
            [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)]
        )
        mp = (xc[:, None] + dx[:, None] * offsets[None, :, 0]).reshape(-1)
        tp = (yc[:, None] + dy[:, None] * offsets[None, :, 1]).reshape(-1)
        point_area = np.repeat(area / 4.0, 4)

        sample0 = self._sample_at(mp, tp)
        jacobian = square_dalitz_jacobian(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        # Package convention is mean(weights * f), hence multiply each physical
        # quadrature weight by the total number of integration points.
        weights = sample0.size * jnp.asarray(point_area) * jacobian
        sample = PhaseSpaceSample(
            s12=sample0.s12,
            s13=sample0.s13,
            s23=sample0.s23,
            weights=weights,
        )
        return AdaptiveSquareDalitzResult(
            sample=sample,
            leaf_bounds=leaf_cells[:, :4].copy(),
            leaf_depths=leaf_cells[:, 4].astype(int),
            leaf_errors=errors,
            mprime=mp,
            thetaprime=tp,
        )

    def sample(self, model, values=None) -> PhaseSpaceSample:
        return self.build(model, values).sample


__all__ = ["AdaptiveSquareDalitzGrid", "AdaptiveSquareDalitzResult"]
