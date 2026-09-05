"""Laura++-style adaptive Gauss--Legendre integration on the Dalitz plot.

The conventional Dalitz integration is performed in the `m13` and `m23`
mass coordinates.  Broad regions use a configurable coarse target width
(default 5 MeV), while bands around explicit narrow resonances are refined
locally.  The default refinement follows the Laura++ `LauIsobarDynamics`
scheme:

* resonances with width <= 20 MeV are considered narrow;
* the refined mass window is m0 +/- 5 Gamma;
* the target bin width in that window is Gamma / 100.

Overlapping narrow bands are resolved by using the finest requested binning.
The tensor product of the resulting one-dimensional partitions automatically
handles crossings between narrow bands.

The returned :class:`~dalitzplotfitter.kinematics.PhaseSpaceSample` follows
the package convention `mean(sample.weights * f)`.  The concatenated raw
Gauss--Legendre weights are therefore multiplied by the total number of
retained physical points.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp
import numpy as np

from dalitzplotfitter.kinematics import PhaseSpaceSample

from .gauss_legendre import _s13_limits_numpy, _scaled_legendre


@dataclass(frozen=True)
class AdaptiveAxisSegment:
    """One non-overlapping integration segment along a Dalitz mass axis."""

    low: float
    high: float
    order: int
    target_width: float
    narrow: bool


@dataclass(frozen=True)
class AdaptiveDalitzGaussLegendreGrid:
    """Piecewise tensor-product Gauss--Legendre grid in `m13` and `m23`.

    Parameters
    ----------
    mother_mass:
        Parent mass in GeV.
    masses:
        Daughter masses `(m1, m2, m3)` in GeV.
    m13_narrow_resonances, m23_narrow_resonances:
        Tuples of `(pole_mass, pole_width)` in GeV.  Entries wider than
        `narrow_width`, non-positive widths, and poles outside the kinematic
        mass range are ignored.
    bin_width:
        Coarse target mass width in GeV.  Laura++ defaults to 5 MeV.
    narrow_width:
        Maximum width treated as narrow.  Laura++ defaults to 20 MeV.
    window_n_widths:
        Half-window around a narrow pole, in units of its width.  Laura++
        defaults to 5.
    binning_factor:
        Narrow-region target spacing is `Gamma / binning_factor`.  Laura++
        defaults to 100.
    """

    mother_mass: float
    masses: tuple[float, float, float]
    m13_narrow_resonances: tuple[tuple[float, float], ...] = ()
    m23_narrow_resonances: tuple[tuple[float, float], ...] = ()
    bin_width: float = 0.005
    narrow_width: float = 0.020
    window_n_widths: float = 5.0
    binning_factor: float = 100.0

    def __post_init__(self) -> None:
        if len(self.masses) != 3:
            raise ValueError(
                "AdaptiveDalitzGaussLegendreGrid requires exactly three daughter masses"
            )
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")
        if self.bin_width <= 0.0:
            raise ValueError("bin_width must be positive")
        if self.narrow_width <= 0.0:
            raise ValueError("narrow_width must be positive")
        if self.window_n_widths <= 0.0:
            raise ValueError("window_n_widths must be positive")
        if self.binning_factor <= 0.0:
            raise ValueError("binning_factor must be positive")

    @property
    def mass_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        m1, m2, m3 = self.masses
        return (
            (m1 + m3, self.mother_mass - m2),
            (m2 + m3, self.mother_mass - m1),
        )

    def _accepted_narrow_resonances(
        self,
        resonances: tuple[tuple[float, float], ...],
        low: float,
        high: float,
    ) -> tuple[tuple[float, float], ...]:
        accepted = []
        for mass, width in resonances:
            mass = float(mass)
            width = float(width)
            if width <= 0.0 or width > self.narrow_width:
                continue
            if mass < low or mass > high:
                continue
            accepted.append((mass, width))
        return tuple(accepted)

    def _axis_segments(
        self,
        low: float,
        high: float,
        resonances: tuple[tuple[float, float], ...],
    ) -> tuple[AdaptiveAxisSegment, ...]:
        narrow = self._accepted_narrow_resonances(resonances, low, high)
        windows: list[tuple[float, float, float]] = []
        boundaries = {float(low), float(high)}

        for mass, width in narrow:
            begin = max(float(low), mass - self.window_n_widths * width)
            end = min(float(high), mass + self.window_n_widths * width)
            if end <= begin:
                continue
            target = width / self.binning_factor
            windows.append((begin, end, target))
            boundaries.add(begin)
            boundaries.add(end)

        sorted_boundaries = sorted(boundaries)
        segments = []
        for begin, end in zip(sorted_boundaries[:-1], sorted_boundaries[1:]):
            if end <= begin:
                continue
            midpoint = 0.5 * (begin + end)
            target = self.bin_width
            is_narrow = False
            for window_begin, window_end, fine_width in windows:
                if window_begin <= midpoint <= window_end:
                    target = min(target, fine_width)
                    is_narrow = True
            order = max(2, int(math.ceil((end - begin) / target)))
            segments.append(
                AdaptiveAxisSegment(
                    low=float(begin),
                    high=float(end),
                    order=order,
                    target_width=float(target),
                    narrow=is_narrow,
                )
            )
        return tuple(segments)

    @property
    def m13_segments(self) -> tuple[AdaptiveAxisSegment, ...]:
        (low, high), _ = self.mass_ranges
        return self._axis_segments(low, high, self.m13_narrow_resonances)

    @property
    def m23_segments(self) -> tuple[AdaptiveAxisSegment, ...]:
        _, (low, high) = self.mass_ranges
        return self._axis_segments(low, high, self.m23_narrow_resonances)

    @property
    def estimated_tensor_points(self) -> int:
        """Number of rectangular tensor-product points before DP masking."""
        n13 = sum(segment.order for segment in self.m13_segments)
        n23 = sum(segment.order for segment in self.m23_segments)
        return int(n13 * n23)

    def sample(self) -> PhaseSpaceSample:
        m1, m2, m3 = self.masses
        invariant_sum = self.mother_mass**2 + m1**2 + m2**2 + m3**2
        s12_min = (m1 + m2) ** 2
        s12_max = (self.mother_mass - m3) ** 2
        scale = max(1.0, self.mother_mass**2)
        tol = 64.0 * np.finfo(float).eps * scale

        s12_parts: list[np.ndarray] = []
        s13_parts: list[np.ndarray] = []
        s23_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []

        for segment13 in self.m13_segments:
            m13_axis, w13_axis = _scaled_legendre(
                segment13.order, segment13.low, segment13.high
            )
            for segment23 in self.m23_segments:
                m23_axis, w23_axis = _scaled_legendre(
                    segment23.order, segment23.low, segment23.high
                )

                m13, m23 = np.meshgrid(m13_axis, m23_axis, indexing="ij")
                w13, w23 = np.meshgrid(w13_axis, w23_axis, indexing="ij")

                s13 = m13 * m13
                s23 = m23 * m23
                s12 = invariant_sum - s13 - s23

                s12_safe = np.clip(s12, s12_min, s12_max)
                low13, high13 = _s13_limits_numpy(
                    s12_safe,
                    mother_mass=self.mother_mass,
                    masses=self.masses,
                )
                physical = (
                    (s12 >= s12_min - tol)
                    & (s12 <= s12_max + tol)
                    & (s13 >= low13 - tol)
                    & (s13 <= high13 + tol)
                )
                if not np.any(physical):
                    continue

                raw_weights = 4.0 * m13 * m23 * w13 * w23
                s12_parts.append(s12[physical].reshape(-1))
                s13_parts.append(s13[physical].reshape(-1))
                s23_parts.append(s23[physical].reshape(-1))
                weight_parts.append(raw_weights[physical].reshape(-1))

        if not s12_parts:
            raise RuntimeError(
                "Adaptive Gauss--Legendre quadrature produced no physical Dalitz points"
            )

        s12 = np.concatenate(s12_parts)
        s13 = np.concatenate(s13_parts)
        s23 = np.concatenate(s23_parts)
        raw_weights = np.concatenate(weight_parts)

        # Package-wide integrators evaluate mean(weights * f).  Convert the
        # concatenated Gauss--Legendre weighted sum to that convention.
        estimator_weights = raw_weights * float(s12.size)

        return PhaseSpaceSample(
            s12=jnp.asarray(s12),
            s13=jnp.asarray(s13),
            s23=jnp.asarray(s23),
            weights=jnp.asarray(estimator_weights),
        )


__all__ = ["AdaptiveAxisSegment", "AdaptiveDalitzGaussLegendreGrid"]
