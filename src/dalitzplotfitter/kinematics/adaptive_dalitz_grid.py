"""Dynamics-aware adaptive quadrature on the physical three-body Dalitz region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from .dalitz_grid import DalitzGrid, dalitz_s13_limits
from .sample import PhaseSpaceSample

Probe = Callable[[Mapping[str, Array]], Array]


@dataclass(frozen=True)
class AdaptiveDalitzGridResult:
    """Adaptive leaf cells and the corresponding weighted midpoint sample."""

    sample: PhaseSpaceSample
    u: Array
    v: Array
    du: Array
    dv: Array
    depth: Array
    score: Array

    @property
    def size(self) -> int:
        return self.sample.size


@dataclass(frozen=True)
class AdaptiveDalitzGrid:
    """Adaptive equal-area-coordinate quadrature for a three-body Dalitz plot.

    The physical Dalitz region is parameterized by the same equal-area mapping
    used by :class:`DalitzGrid`.  Adaptivity is performed in the auxiliary
    ``(u, v)`` square, whose Jacobian to ``(s12, s13)`` is the constant Dalitz
    area.  Cells can therefore be recursively split without any special
    treatment of the curved physical boundary.

    Refinement is deliberately agnostic about the type of dynamics.  Each
    supplied ``probe`` is simply a function on Dalitz points and may represent
    a Breit-Wigner, a dispersive amplitude, a spline, a K-matrix term, or any
    future model.  For complex probes the refinement estimator is applied to
    ``abs(probe)**2``.

    A cell is refined when either the variation across its centre/quarter
    points or the discrepancy between the centre estimate and the four-child
    midpoint estimate exceeds ``tolerance``.  ``base_resolution`` is therefore
    also the discovery scale: features much narrower than every initial cell
    can only be found if at least one probe point samples them.
    """

    mother_mass: float
    masses: tuple[float, float, float]
    base_resolution: int = 32
    max_depth: int = 4
    tolerance: float = 0.05
    absolute_floor: float = 1e-14
    boundary_resolution: int | None = None
    max_cells: int = 2_000_000

    def __post_init__(self) -> None:
        if self.base_resolution < 2:
            raise ValueError("base_resolution must be at least 2")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.absolute_floor <= 0.0:
            raise ValueError("absolute_floor must be positive")
        if self.max_cells < self.base_resolution**2:
            raise ValueError("max_cells must accommodate the initial grid")
        # Reuse DalitzGrid validation and area mapping convention.
        DalitzGrid(
            self.mother_mass,
            self.masses,
            resolution=max(2, self.base_resolution),
            boundary_resolution=self.boundary_resolution,
        )

    def _mapping(self):
        grid = DalitzGrid(
            self.mother_mass,
            self.masses,
            resolution=max(2, self.base_resolution),
            boundary_resolution=self.boundary_resolution,
        )
        support, cumulative, area = grid._area_mapping()
        return support, cumulative, area

    def _map_uv(self, u: Array, v: Array, support: Array, cumulative: Array, area: Array):
        u = jnp.asarray(u)
        v = jnp.asarray(v)
        s12 = jnp.interp(u * area, cumulative, support)
        low, high = dalitz_s13_limits(
            s12,
            mother_mass=self.mother_mass,
            masses=self.masses,
        )
        s13 = low + (high - low) * v
        m1, m2, m3 = self.masses
        s23 = (
            self.mother_mass**2
            + m1**2
            + m2**2
            + m3**2
            - s12
            - s13
        )
        return {"s12": s12, "s13": s13, "s23": s23}

    def _probe_score(
        self,
        probe: Probe,
        u0: Array,
        v0: Array,
        du: Array,
        dv: Array,
        support: Array,
        cumulative: Array,
        area: Array,
    ) -> Array:
        # Cell centre plus the four would-be child midpoints.  Comparing these
        # estimates gives a local quadrature-error signal while the spread gives
        # a direct local-resolution signal.
        u = jnp.stack(
            (
                u0,
                u0 - 0.25 * du,
                u0 - 0.25 * du,
                u0 + 0.25 * du,
                u0 + 0.25 * du,
            ),
            axis=1,
        )
        v = jnp.stack(
            (
                v0,
                v0 - 0.25 * dv,
                v0 + 0.25 * dv,
                v0 - 0.25 * dv,
                v0 + 0.25 * dv,
            ),
            axis=1,
        )
        flat = self._map_uv(
            u.reshape(-1), v.reshape(-1), support, cumulative, area
        )
        values = jnp.asarray(probe(flat)).reshape(u.shape)
        density = jnp.abs(values) ** 2 if jnp.iscomplexobj(values) else jnp.abs(values)

        centre = density[:, 0]
        children = density[:, 1:]
        child_mean = jnp.mean(children, axis=1)
        local_scale = jnp.maximum(jnp.mean(density, axis=1), self.absolute_floor)
        variation = (jnp.max(density, axis=1) - jnp.min(density, axis=1)) / local_scale
        quadrature_error = jnp.abs(child_mean - centre) / local_scale
        return jnp.maximum(variation, quadrature_error)

    def build(self, probes: Sequence[Probe]) -> AdaptiveDalitzGridResult:
        """Build an adaptive grid from one or more arbitrary dynamics probes."""

        probes = tuple(probes)
        if not probes:
            raise ValueError("AdaptiveDalitzGrid requires at least one probe")

        support, cumulative, area = self._mapping()
        n = int(self.base_resolution)
        coords = (jnp.arange(n, dtype=support.dtype) + 0.5) / n
        uu, vv = jnp.meshgrid(coords, coords, indexing="ij")
        u = uu.reshape(-1)
        v = vv.reshape(-1)
        du = jnp.full_like(u, 1.0 / n)
        dv = jnp.full_like(v, 1.0 / n)
        depth = jnp.zeros_like(u, dtype=jnp.int32)
        score = jnp.zeros_like(u)

        finished: list[tuple[Array, Array, Array, Array, Array, Array]] = []

        for level in range(self.max_depth + 1):
            scores = jnp.stack(
                [
                    self._probe_score(
                        probe, u, v, du, dv, support, cumulative, area
                    )
                    for probe in probes
                ],
                axis=0,
            )
            score = jnp.max(scores, axis=0)

            if level == self.max_depth:
                finished.append((u, v, du, dv, depth, score))
                break

            refine = score > self.tolerance
            keep = ~refine
            if bool(jnp.any(keep)):
                finished.append(
                    (u[keep], v[keep], du[keep], dv[keep], depth[keep], score[keep])
                )
            if not bool(jnp.any(refine)):
                break

            ur = u[refine]
            vr = v[refine]
            dur = du[refine] * 0.5
            dvr = dv[refine] * 0.5
            dr = depth[refine] + 1

            offsets_u = jnp.asarray((-0.5, -0.5, 0.5, 0.5), dtype=u.dtype)
            offsets_v = jnp.asarray((-0.5, 0.5, -0.5, 0.5), dtype=v.dtype)
            u = (ur[:, None] + offsets_u[None, :] * dur[:, None]).reshape(-1)
            v = (vr[:, None] + offsets_v[None, :] * dvr[:, None]).reshape(-1)
            du = jnp.repeat(dur, 4)
            dv = jnp.repeat(dvr, 4)
            depth = jnp.repeat(dr, 4)
            score = jnp.zeros_like(u)

            finished_count = sum(int(block[0].size) for block in finished)
            if finished_count + int(u.size) > self.max_cells:
                raise RuntimeError(
                    "adaptive grid exceeded max_cells; increase tolerance, reduce "
                    "max_depth/base_resolution, or raise max_cells"
                )

        u = jnp.concatenate([block[0] for block in finished])
        v = jnp.concatenate([block[1] for block in finished])
        du = jnp.concatenate([block[2] for block in finished])
        dv = jnp.concatenate([block[3] for block in finished])
        depth = jnp.concatenate([block[4] for block in finished])
        score = jnp.concatenate([block[5] for block in finished])

        data = self._map_uv(u, v, support, cumulative, area)
        # Package convention is mean(weights * f).  A leaf midpoint represents
        # physical area A_DP * du * dv, hence weight_i = Nleaf * that area.
        weights = u.size * area * du * dv
        sample = PhaseSpaceSample(
            s12=data["s12"],
            s13=data["s13"],
            s23=data["s23"],
            weights=weights,
        )
        return AdaptiveDalitzGridResult(sample, u, v, du, dv, depth, score)

    def sample(self, probes: Sequence[Probe]) -> PhaseSpaceSample:
        """Return only the weighted phase-space sample for integration/fitting."""

        return self.build(probes).sample
