"""Two-dimensional quasi-model-independent Dalitz amplitude."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp


def _catmull_rom(p0, p1, p2, p3, t):
    """Local cubic interpolation through p1 and p2."""

    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t**3
    )


def _indices_and_fraction(x, centers):
    """Neighbouring center indices and interpolation fraction."""

    n = centers.shape[0]
    right = jnp.searchsorted(centers, x, side="right")
    left = jnp.clip(right - 1, 0, n - 2)
    right = left + 1
    x0 = centers[left]
    x1 = centers[right]
    t = jnp.where(x1 > x0, (x - x0) / (x1 - x0), 0.0)
    return left, right, jnp.clip(t, 0.0, 1.0)


def _bilinear(x, y, xcenters, ycenters, values):
    ix0, ix1, tx = _indices_and_fraction(x, xcenters)
    iy0, iy1, ty = _indices_and_fraction(y, ycenters)
    v00 = values[ix0, iy0]
    v10 = values[ix1, iy0]
    v01 = values[ix0, iy1]
    v11 = values[ix1, iy1]
    vx0 = (1.0 - tx) * v00 + tx * v10
    vx1 = (1.0 - tx) * v01 + tx * v11
    return (1.0 - ty) * vx0 + ty * vx1


def _bicubic_one(x, y, xcenters, ycenters, values):
    """Tensor-product local Catmull-Rom interpolation for one point."""

    nx, ny = values.shape
    ix1, ix2, tx = _indices_and_fraction(x, xcenters)
    iy1, iy2, ty = _indices_and_fraction(y, ycenters)
    ix = jnp.clip(jnp.asarray([ix1 - 1, ix1, ix2, ix2 + 1]), 0, nx - 1)
    iy = jnp.clip(jnp.asarray([iy1 - 1, iy1, iy2, iy2 + 1]), 0, ny - 1)
    patch = values[ix[:, None], iy[None, :]]
    along_x = jax.vmap(lambda column: _catmull_rom(column[0], column[1], column[2], column[3], tx), in_axes=1)(patch)
    return _catmull_rom(along_x[0], along_x[1], along_x[2], along_x[3], ty)


@dataclass(frozen=True)
class QMI2D:
    """Complex amplitude field defined bin-by-bin over the Dalitz plane.

    ``magnitudes`` and ``phases`` are arrays with shape
    ``(len(s12_edges)-1, len(s13_edges)-1)``. Each cell therefore owns one
    complex value ``a_ij exp(i phi_ij)``. Entries may be numerical values or fit
    ``Parameter`` objects.

    Interpolation modes:

    - ``none``: piecewise-constant complex amplitude per bin;
    - ``linear``: bilinear interpolation of magnitude and phase between bin centers;
    - ``cubic``: local bicubic Catmull-Rom interpolation of magnitude and phase.

    With ``folded=True`` the coordinates are replaced by
    ``(min(s12,s13), max(s12,s13))`` before lookup/interpolation. This is useful
    for final states with two identical particles, such as D_s+ -> pi- pi+ pi+.
    """

    s12_edges: tuple[float, ...]
    s13_edges: tuple[float, ...]
    magnitudes: tuple[tuple[object, ...], ...]
    phases: tuple[tuple[object, ...], ...]
    interpolation: str = "none"
    folded: bool = False

    def __post_init__(self) -> None:
        if len(self.s12_edges) < 2 or len(self.s13_edges) < 2:
            raise ValueError("QMI2D requires at least one bin on each axis")
        if any(b <= a for a, b in zip(self.s12_edges[:-1], self.s12_edges[1:])):
            raise ValueError("QMI2D s12_edges must be strictly increasing")
        if any(b <= a for a, b in zip(self.s13_edges[:-1], self.s13_edges[1:])):
            raise ValueError("QMI2D s13_edges must be strictly increasing")
        nx = len(self.s12_edges) - 1
        ny = len(self.s13_edges) - 1
        if len(self.magnitudes) != nx or any(len(row) != ny for row in self.magnitudes):
            raise ValueError("QMI2D magnitudes shape must match the 2D binning")
        if len(self.phases) != nx or any(len(row) != ny for row in self.phases):
            raise ValueError("QMI2D phases shape must match the 2D binning")
        if self.interpolation not in {"none", "linear", "cubic"}:
            raise ValueError("QMI2D interpolation must be 'none', 'linear', or 'cubic'")
        if self.interpolation == "cubic" and (nx < 2 or ny < 2):
            raise ValueError("cubic QMI2D interpolation requires at least 2x2 bins")

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.s12_edges) - 1, len(self.s13_edges) - 1)

    def _coordinates(self, data: Mapping[str, object]):
        s12 = jnp.asarray(data["s12"])
        s13 = jnp.asarray(data["s13"])
        if self.folded:
            return jnp.minimum(s12, s13), jnp.maximum(s12, s13)
        return s12, s13

    def interpolated_magnitude_phase(self, data: Mapping[str, object]):
        x, y = self._coordinates(data)
        xedges = jnp.asarray(self.s12_edges, dtype=x.dtype)
        yedges = jnp.asarray(self.s13_edges, dtype=y.dtype)
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        magnitudes = jnp.asarray(self.magnitudes, dtype=x.dtype)
        phases = jnp.asarray(self.phases, dtype=x.dtype)

        if self.interpolation == "none":
            ix = jnp.clip(jnp.searchsorted(xedges, x, side="right") - 1, 0, magnitudes.shape[0] - 1)
            iy = jnp.clip(jnp.searchsorted(yedges, y, side="right") - 1, 0, magnitudes.shape[1] - 1)
            return magnitudes[ix, iy], phases[ix, iy]

        x_eval = jnp.clip(x, xcenters[0], xcenters[-1])
        y_eval = jnp.clip(y, ycenters[0], ycenters[-1])
        if self.interpolation == "linear":
            return (
                _bilinear(x_eval, y_eval, xcenters, ycenters, magnitudes),
                _bilinear(x_eval, y_eval, xcenters, ycenters, phases),
            )

        cubic = jax.vmap(_bicubic_one, in_axes=(0, 0, None, None, None))
        xflat = jnp.ravel(x_eval)
        yflat = jnp.ravel(y_eval)
        mag = cubic(xflat, yflat, xcenters, ycenters, magnitudes).reshape(x_eval.shape)
        phase = cubic(xflat, yflat, xcenters, ycenters, phases).reshape(x_eval.shape)
        return mag, phase

    def __call__(self, data: Mapping[str, object], parameters=None):
        del parameters
        magnitude, phase = self.interpolated_magnitude_phase(data)
        return magnitude * jnp.exp(1j * phase)


__all__ = ["QMI2D"]
