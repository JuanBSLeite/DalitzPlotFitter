"""Physics-informed adaptive quadrature in Square-Dalitz coordinates.

This grid keeps the user-selected Square-Dalitz coordinates but refines only
cells crossed by declared narrow-resonance bands. The broad region retains an
effective resolution comparable to SquareDalitzGrid.

Refinement targets the Laura++ narrow-resonance convention: a band is relevant
inside m0 +/- 5 Gamma and the local Gauss-Legendre node spacing is driven toward
Gamma/100 by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.kinematics import (
    PhaseSpaceSample,
    square_dalitz_jacobian,
    square_dalitz_to_invariants,
)


_PAIR_KEY = {
    (0, 1): 0,
    (0, 2): 1,
    (1, 2): 2,
}


def _sorted_pair(pair: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted(pair))


def _kallen(x, y, z):
    return x**2 + y**2 + z**2 - 2.0 * x * y - 2.0 * x * z - 2.0 * y * z


def _square_to_invariants_numpy(
    mprime,
    thetaprime,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
    pair: tuple[int, int],
):
    """NumPy version used only while constructing the adaptive cell layout."""

    i, j = pair
    k = next(index for index in range(3) if index not in pair)
    mi, mj, mk = masses[i], masses[j], masses[k]

    mp = np.asarray(mprime, dtype=np.float64)
    tp = np.asarray(thetaprime, dtype=np.float64)

    m_min = mi + mj
    m_max = mother_mass - mk
    delta_m = m_max - m_min

    m_ij = m_min + 0.5 * delta_m * (1.0 + np.cos(np.pi * mp))
    s_ij = m_ij**2
    theta = np.pi * tp

    e_i = (s_ij + mi**2 - mj**2) / (2.0 * m_ij)
    e_k = (mother_mass**2 - s_ij - mk**2) / (2.0 * m_ij)
    q = np.sqrt(np.maximum(_kallen(s_ij, mi**2, mj**2), 0.0)) / (2.0 * m_ij)
    p = np.sqrt(
        np.maximum(_kallen(mother_mass**2, s_ij, mk**2), 0.0)
    ) / (2.0 * m_ij)

    s_ik = mi**2 + mk**2 + 2.0 * (e_i * e_k - q * p * np.cos(theta))
    total = mother_mass**2 + sum(value**2 for value in masses)
    s_jk = total - s_ij - s_ik

    values = {
        _sorted_pair((i, j)): s_ij,
        _sorted_pair((i, k)): s_ik,
        _sorted_pair((j, k)): s_jk,
    }
    return (
        values[(0, 1)],
        values[(0, 2)],
        values[(1, 2)],
    )


@dataclass(frozen=True)
class AdaptiveSquareDalitzGrid:
    """Locally refined Square-Dalitz Gauss-Legendre quadrature."""

    mother_mass: float
    masses: tuple[float, float, float]
    narrow_resonances: tuple[tuple[tuple[int, int], float, float], ...]
    resolution: int = 1000
    pair: tuple[int, int] = (0, 1)
    window_n_widths: float = 5.0
    binning_factor: float = 100.0
    cell_order: int = 8
    max_depth: int = 12

    def __post_init__(self) -> None:
        if self.resolution < 2:
            raise ValueError("resolution must be at least 2")
        if len(self.masses) != 3:
            raise ValueError("exactly three daughter masses are required")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")
        i, j = self.pair
        if i == j or i not in (0, 1, 2) or j not in (0, 1, 2):
            raise ValueError("pair must contain two distinct indices from 0, 1, 2")
        if self.window_n_widths <= 0.0:
            raise ValueError("window_n_widths must be positive")
        if self.binning_factor <= 0.0:
            raise ValueError("binning_factor must be positive")
        if self.cell_order < 2:
            raise ValueError("cell_order must be at least 2")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")

    def _cell_mass_ranges(
        self,
        mp_low: float,
        mp_high: float,
        tp_low: float,
        tp_high: float,
    ) -> tuple[tuple[float, float], ...]:
        mp_mid = 0.5 * (mp_low + mp_high)
        tp_mid = 0.5 * (tp_low + tp_high)
        mp = np.asarray(
            [
                mp_low, mp_low, mp_low,
                mp_mid, mp_mid, mp_mid,
                mp_high, mp_high, mp_high,
            ]
        )
        tp = np.asarray(
            [
                tp_low, tp_mid, tp_high,
                tp_low, tp_mid, tp_high,
                tp_low, tp_mid, tp_high,
            ]
        )
        invariants = _square_to_invariants_numpy(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        ranges = []
        for invariant in invariants:
            mass = np.sqrt(np.maximum(invariant, 0.0))
            ranges.append((float(np.min(mass)), float(np.max(mass))))
        return tuple(ranges)

    def _needs_refinement(
        self,
        mp_low: float,
        mp_high: float,
        tp_low: float,
        tp_high: float,
    ) -> bool:
        ranges = self._cell_mass_ranges(mp_low, mp_high, tp_low, tp_high)

        for pair, pole_mass, pole_width in self.narrow_resonances:
            if pole_width <= 0.0:
                continue
            index = _PAIR_KEY[_sorted_pair(pair)]
            mass_low, mass_high = ranges[index]
            window_low = pole_mass - self.window_n_widths * pole_width
            window_high = pole_mass + self.window_n_widths * pole_width
            if mass_high < window_low or mass_low > window_high:
                continue

            allowed_span = self.cell_order * pole_width / self.binning_factor
            if mass_high - mass_low > allowed_span:
                return True

        return False

    def _leaf_cells(self) -> list[tuple[float, float, float, float]]:
        base_cells = max(1, int(math.ceil(self.resolution / self.cell_order)))
        edges = np.linspace(0.0, 1.0, base_cells + 1)

        stack: list[tuple[float, float, float, float, int]] = []
        for i in range(base_cells):
            for j in range(base_cells):
                stack.append(
                    (edges[i], edges[i + 1], edges[j], edges[j + 1], 0)
                )

        leaves: list[tuple[float, float, float, float]] = []
        while stack:
            mp_low, mp_high, tp_low, tp_high, depth = stack.pop()
            if (
                depth < self.max_depth
                and self._needs_refinement(mp_low, mp_high, tp_low, tp_high)
            ):
                mp_mid = 0.5 * (mp_low + mp_high)
                tp_mid = 0.5 * (tp_low + tp_high)
                next_depth = depth + 1
                stack.extend(
                    [
                        (mp_low, mp_mid, tp_low, tp_mid, next_depth),
                        (mp_low, mp_mid, tp_mid, tp_high, next_depth),
                        (mp_mid, mp_high, tp_low, tp_mid, next_depth),
                        (mp_mid, mp_high, tp_mid, tp_high, next_depth),
                    ]
                )
            else:
                leaves.append((mp_low, mp_high, tp_low, tp_high))
        return leaves

    @property
    def leaf_cell_count(self) -> int:
        return len(self._leaf_cells())

    def sample(self) -> PhaseSpaceSample:
        x, w = np.polynomial.legendre.leggauss(self.cell_order)
        leaves = self._leaf_cells()

        mp_parts: list[np.ndarray] = []
        tp_parts: list[np.ndarray] = []
        raw_weight_parts: list[np.ndarray] = []

        for mp_low, mp_high, tp_low, tp_high in leaves:
            mp_nodes = 0.5 * (mp_high - mp_low) * x + 0.5 * (mp_high + mp_low)
            tp_nodes = 0.5 * (tp_high - tp_low) * x + 0.5 * (tp_high + tp_low)
            mp_weights = 0.5 * (mp_high - mp_low) * w
            tp_weights = 0.5 * (tp_high - tp_low) * w

            mp, tp = np.meshgrid(mp_nodes, tp_nodes, indexing="ij")
            wm, wt = np.meshgrid(mp_weights, tp_weights, indexing="ij")

            mp_parts.append(mp.reshape(-1))
            tp_parts.append(tp.reshape(-1))
            raw_weight_parts.append((wm * wt).reshape(-1))

        mp = np.concatenate(mp_parts)
        tp = np.concatenate(tp_parts)
        raw_weights = np.concatenate(raw_weight_parts)

        mp_jax = jnp.asarray(mp)
        tp_jax = jnp.asarray(tp)
        s12, s13, s23 = square_dalitz_to_invariants(
            mp_jax,
            tp_jax,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        jacobian = square_dalitz_jacobian(
            mp_jax,
            tp_jax,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )

        weights = jacobian * jnp.asarray(raw_weights * float(mp.size))
        return PhaseSpaceSample(s12=s12, s13=s13, s23=s23, weights=weights)


__all__ = ["AdaptiveSquareDalitzGrid"]
