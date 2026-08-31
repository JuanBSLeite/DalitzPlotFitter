"""Adaptive ordinary-Dalitz integration on the equal-area (u, v) map."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.kinematics.dalitz_grid import DalitzGrid, dalitz_s13_limits
from dalitzplotfitter.kinematics.sample import PhaseSpaceSample


@dataclass(frozen=True)
class AdaptiveDalitzResult:
    """Adaptive ordinary-Dalitz normalization sample and leaf diagnostics."""

    sample: PhaseSpaceSample
    leaf_bounds: np.ndarray
    leaf_depths: np.ndarray
    leaf_errors: np.ndarray
    u: np.ndarray
    v: np.ndarray

    @property
    def size(self) -> int:
        return self.sample.size

    @property
    def n_leaves(self) -> int:
        return int(self.leaf_bounds.shape[0])


@dataclass(frozen=True)
class AdaptiveDalitzGrid:
    """Adaptive integration using the same equal-area map as :class:`DalitzGrid`.

    The ordinary Dalitz grid internally maps a unit square ``(u,v)`` into the
    physical ``(s12,s13)`` region with constant Jacobian equal to the total
    Dalitz area. Refinement is therefore performed in ``(u,v)`` while every
    generated point remains physical.

    As in ``AdaptiveSquareDalitzGrid``, refinement is driven by convergence of
    the full raw bilinear matrix ``F_i^* F_j`` and is independent of resonance
    metadata such as masses or widths.
    """

    mother_mass: float
    masses: tuple[float, float, float]
    base_resolution: int = 24
    min_depth: int = 1
    max_depth: int = 5
    tolerance: float = 0.02
    matrix_floor: float = 1e-8
    max_cells: int = 200_000
    boundary_resolution: int | None = None

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
        if len(self.masses) != 3:
            raise ValueError("AdaptiveDalitzGrid requires exactly three daughter masses")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")

    def _mapping(self):
        helper = DalitzGrid(
            self.mother_mass,
            self.masses,
            resolution=max(self.base_resolution, 2),
            boundary_resolution=self.boundary_resolution,
        )
        support, cumulative, area = helper._area_mapping()
        return np.asarray(support), np.asarray(cumulative), float(area)

    def _sample_at(self, u: np.ndarray, v: np.ndarray, mapping) -> PhaseSpaceSample:
        support, cumulative, area = mapping
        uu = np.asarray(u, dtype=np.float64)
        vv = np.asarray(v, dtype=np.float64)
        s12 = np.interp(uu * area, cumulative, support)
        low, high = dalitz_s13_limits(
            jnp.asarray(s12),
            mother_mass=self.mother_mass,
            masses=self.masses,
        )
        low = np.asarray(low)
        high = np.asarray(high)
        s13 = low + (high - low) * vv
        m1, m2, m3 = self.masses
        total = self.mother_mass**2 + m1**2 + m2**2 + m3**2
        s23 = total - s12 - s13
        return PhaseSpaceSample(
            s12=jnp.asarray(s12),
            s13=jnp.asarray(s13),
            s23=jnp.asarray(s23),
            weights=jnp.ones_like(jnp.asarray(s12)),
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

    def _cell_errors(self, model, cells: np.ndarray, mapping, values=None) -> np.ndarray:
        u0, u1, v0, v1 = (cells[:, i] for i in range(4))
        du = u1 - u0
        dv = v1 - v0
        cell_area = du * dv
        uc = 0.5 * (u0 + u1)
        vc = 0.5 * (v0 + v1)

        center = self._sample_at(uc, vc, mapping)
        fc = self._raw_components(model, center, values)
        bc = np.conj(fc)[:, :, None] * fc[:, None, :]
        coarse = cell_area[:, None, None] * bc

        offsets = np.asarray(
            [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)]
        )
        uf = (uc[:, None] + du[:, None] * offsets[None, :, 0]).reshape(-1)
        vf = (vc[:, None] + dv[:, None] * offsets[None, :, 1]).reshape(-1)
        fine_sample = self._sample_at(uf, vf, mapping)
        ff = self._raw_components(model, fine_sample, values)
        bf = np.conj(ff)[:, :, None] * ff[:, None, :]
        n_cells = cells.shape[0]
        bf = bf.reshape(n_cells, 4, bf.shape[1], bf.shape[2])
        fine = cell_area[:, None, None] * np.mean(bf, axis=1)

        magnitude = np.maximum(np.abs(fine), np.abs(coarse))
        local_scale = np.max(magnitude, axis=(1, 2), keepdims=True)
        relevant = magnitude >= self.matrix_floor * np.maximum(local_scale, 1e-300)
        denominator = np.maximum(magnitude, self.matrix_floor * np.maximum(local_scale, 1e-300))
        relative = np.where(relevant, np.abs(fine - coarse) / denominator, 0.0)
        return np.max(relative, axis=(1, 2))

    @staticmethod
    def _split(cells: np.ndarray) -> np.ndarray:
        children = []
        for u0, u1, v0, v1, depth in cells:
            um = 0.5 * (u0 + u1)
            vm = 0.5 * (v0 + v1)
            next_depth = depth + 1
            children.extend(
                [
                    (u0, um, v0, vm, next_depth),
                    (um, u1, v0, vm, next_depth),
                    (u0, um, vm, v1, next_depth),
                    (um, u1, vm, v1, next_depth),
                ]
            )
        return np.asarray(children, dtype=np.float64)

    def build(self, model, values=None) -> AdaptiveDalitzResult:
        mapping = self._mapping()
        n = self.base_resolution
        edges = np.linspace(0.0, 1.0, n + 1)
        initial = [
            (edges[i], edges[i + 1], edges[j], edges[j + 1], 0.0)
            for i in range(n)
            for j in range(n)
        ]
        active = np.asarray(initial, dtype=np.float64)
        leaves: list[np.ndarray] = []
        leaf_errors: list[np.ndarray] = []

        while active.size:
            if active.shape[0] + sum(item.shape[0] for item in leaves) > self.max_cells:
                raise RuntimeError("adaptive Dalitz grid exceeded max_cells")
            errors = self._cell_errors(model, active, mapping, values)
            depth = active[:, 4].astype(int)
            refine = ((depth < self.min_depth) | (errors > self.tolerance)) & (depth < self.max_depth)
            keep = ~refine
            if np.any(keep):
                leaves.append(active[keep])
                leaf_errors.append(errors[keep])
            active = self._split(active[refine]) if np.any(refine) else np.empty((0, 5))

        leaf_cells = np.concatenate(leaves, axis=0)
        errors = np.concatenate(leaf_errors, axis=0)
        u0, u1, v0, v1 = (leaf_cells[:, i] for i in range(4))
        du = u1 - u0
        dv = v1 - v0
        uv_area = du * dv
        uc = 0.5 * (u0 + u1)
        vc = 0.5 * (v0 + v1)
        offsets = np.asarray(
            [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)]
        )
        u = (uc[:, None] + du[:, None] * offsets[None, :, 0]).reshape(-1)
        v = (vc[:, None] + dv[:, None] * offsets[None, :, 1]).reshape(-1)
        point_uv_area = np.repeat(uv_area / 4.0, 4)

        sample0 = self._sample_at(u, v, mapping)
        physical_area = mapping[2]
        physical_weights = physical_area * point_uv_area
        weights = sample0.size * jnp.asarray(physical_weights)
        sample = PhaseSpaceSample(
            s12=sample0.s12,
            s13=sample0.s13,
            s23=sample0.s23,
            weights=weights,
        )
        return AdaptiveDalitzResult(
            sample=sample,
            leaf_bounds=leaf_cells[:, :4].copy(),
            leaf_depths=leaf_cells[:, 4].astype(int),
            leaf_errors=errors,
            u=u,
            v=v,
        )

    def sample(self, model, values=None) -> PhaseSpaceSample:
        return self.build(model, values).sample


__all__ = ["AdaptiveDalitzGrid", "AdaptiveDalitzResult"]
