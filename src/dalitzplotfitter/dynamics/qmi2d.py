"""Two-dimensional quasi-model-independent Dalitz amplitude."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import isfinite, sqrt

import jax
import jax.numpy as jnp

from dalitzplotfitter.fit.parameters import Parameter, ParameterKind


def _kallen(x, y, z):
    return x*x + y*y + z*z - 2.0*x*y - 2.0*x*z - 2.0*y*z


def _s13_limits_scalar(s12, mother_mass, masses):
    m1, m2, m3 = masses
    if s12 == 0.0 and m1 == 0.0 and m2 == 0.0:
        return m3 * m3, mother_mass * mother_mass
    root = sqrt(max(s12, 0.0))
    e1 = (s12 + m1*m1 - m2*m2) / (2.0*root)
    e3 = (mother_mass*mother_mass - s12 - m3*m3) / (2.0*root)
    q = sqrt(max(_kallen(s12, m1*m1, m2*m2), 0.0)) / (2.0*root)
    p = sqrt(max(_kallen(mother_mass*mother_mass, s12, m3*m3), 0.0)) / (2.0*root)
    common = m1*m1 + m3*m3 + 2.0*e1*e3
    spread = 2.0*q*p
    return common-spread, common+spread


def physical_bin_mask(s12_edges, s13_edges, *, mother_mass, masses, folded=False, samples_per_bin=129):
    """Estimate bin intersections by sampling the analytic Dalitz boundary.

    Each bin is clipped to the physical s12 range before sampling. Finite
    sampling can miss very narrow intersections; increase samples_per_bin
    to check boundary-mask convergence.
    """
    if isinstance(samples_per_bin, bool) or not isinstance(samples_per_bin, int):
        raise ValueError("samples_per_bin must be an integer")
    if samples_per_bin < 3:
        raise ValueError("samples_per_bin must be at least 3")
    xedges = tuple(float(v) for v in s12_edges)
    yedges = tuple(float(v) for v in s13_edges)
    if len(masses) != 3:
        raise ValueError("masses must contain three daughter masses")
    masses = tuple(float(v) for v in masses)
    mother_mass = float(mother_mass)
    if (not all(isfinite(v) and v >= 0 for v in masses)
            or not isfinite(mother_mass) or mother_mass <= sum(masses)):
        raise ValueError("Masses must be finite and the parent above threshold")
    for edges in (xedges, yedges):
        if (len(edges) < 2 or not all(isfinite(v) for v in edges)
                or any(b <= a for a, b in zip(edges[:-1], edges[1:]))):
            raise ValueError("Bin edges must be finite and strictly increasing")
    physical_low = (masses[0] + masses[1]) ** 2
    physical_high = (mother_mass - masses[2]) ** 2
    rows = []
    for x0, x1 in zip(xedges[:-1], xedges[1:]):
        x0, x1 = max(x0, physical_low), min(x1, physical_high)
        if x0 > x1:
            rows.append(tuple(False for _ in yedges[1:]))
            continue
        row = []
        for y0, y1 in zip(yedges[:-1], yedges[1:]):
            active = False
            for k in range(samples_per_bin):
                t = k/(samples_per_bin-1)
                x = x0 + t*(x1-x0)
                low, high = _s13_limits_scalar(x, mother_mass, masses)
                lower = max(y0, x) if folded else y0
                if y1 >= lower and high >= lower and low <= y1:
                    active = True
                    break
            row.append(active)
        rows.append(tuple(row))
    return tuple(rows)


def _catmull_rom(p0, p1, p2, p3, t):
    return 0.5*(2.0*p1 + (-p0+p2)*t + (2.0*p0-5.0*p1+4.0*p2-p3)*t**2 + (-p0+3.0*p1-3.0*p2+p3)*t**3)


def _indices_and_fraction(x, centers):
    n = centers.shape[0]
    right = jnp.searchsorted(centers, x, side="right")
    left = jnp.clip(right-1, 0, n-2)
    right = left+1
    x0, x1 = centers[left], centers[right]
    t = jnp.where(x1 > x0, (x-x0)/(x1-x0), 0.0)
    return left, right, jnp.clip(t, 0.0, 1.0)


def _bilinear(x, y, xc, yc, values):
    ix0, ix1, tx = _indices_and_fraction(x, xc)
    iy0, iy1, ty = _indices_and_fraction(y, yc)
    v00, v10 = values[ix0,iy0], values[ix1,iy0]
    v01, v11 = values[ix0,iy1], values[ix1,iy1]
    return (1.0-ty)*((1.0-tx)*v00+tx*v10) + ty*((1.0-tx)*v01+tx*v11)


def _cubic_axis(values, centers, left, right, fraction):
    """Hermite interpolation with slopes in physical coordinates.

    Ghost coordinates extend one interval past each edge with a repeated
    value, preserving the previous Catmull-Rom behavior on uniform grids.
    """
    before = jnp.maximum(left - 1, 0)
    after = jnp.minimum(right + 1, centers.shape[0] - 1)
    x1, x2 = centers[left], centers[right]
    x0 = jnp.where(left == 0, 2 * x1 - x2, centers[before])
    x3 = jnp.where(right == centers.shape[0] - 1, 2 * x2 - x1, centers[after])
    p0, p1, p2, p3 = values[before], values[left], values[right], values[after]
    width = x2 - x1
    slope1 = (p2 - p0) / (x2 - x0)
    slope2 = (p3 - p1) / (x3 - x1)
    t = fraction
    return ((2*t**3 - 3*t**2 + 1)*p1 + (t**3 - 2*t**2 + t)*width*slope1
            + (-2*t**3 + 3*t**2)*p2 + (t**3 - t**2)*width*slope2)


def _bicubic_one(x, y, xc, yc, values):
    ix1, ix2, tx = _indices_and_fraction(x, xc)
    iy1, iy2, ty = _indices_and_fraction(y, yc)
    # Retain local 4x4 support, including for large event samples.
    ix = jnp.clip(jnp.asarray([ix1-1, ix1, ix2, ix2+1]), 0, values.shape[0]-1)
    x1, x2 = xc[ix1], xc[ix2]
    x0 = jnp.where(ix1 == 0, 2*x1-x2, xc[ix[0]])
    x3 = jnp.where(ix2 == xc.shape[0]-1, 2*x2-x1, xc[ix[3]])
    xcoords = jnp.stack((x0, x1, x2, x3))
    iy = jnp.clip(jnp.asarray([iy1-1, iy1, iy2, iy2+1]), 0, values.shape[1]-1)
    patch = values[ix[:, None], iy[None, :]]
    along = jax.vmap(lambda c: _cubic_axis(c, xcoords, 1, 2, tx), in_axes=1)(patch)
    y1, y2 = yc[iy1], yc[iy2]
    y0 = jnp.where(iy1 == 0, 2*y1-y2, yc[iy[0]])
    y3 = jnp.where(iy2 == yc.shape[0]-1, 2*y2-y1, yc[iy[3]])
    return _cubic_axis(along, jnp.stack((y0, y1, y2, y3)), 1, 2, ty)


def _nearest_active_sources(mask):
    """Return a fixed flat gather map used to fill inactive interpolation cells."""
    if mask is None:
        return None
    active = [(i, j) for i, row in enumerate(mask) for j, ok in enumerate(row) if ok]
    if not active:
        raise ValueError("QMI2D active_mask contains no physical bins")
    ny = len(mask[0])
    sources = []
    for i, row in enumerate(mask):
        for j, ok in enumerate(row):
            if ok:
                ai, aj = i, j
            else:
                ai, aj = min(active, key=lambda p: (p[0]-i)**2 + (p[1]-j)**2)
            sources.append(ai * ny + aj)
    return tuple(sources)


def _fill_inactive_from_sources(values, sources):
    if sources is None:
        return values
    flat = jnp.ravel(values)
    return flat[jnp.asarray(sources, dtype=jnp.int32)].reshape(values.shape)


@dataclass(frozen=True)
class QMI2D:
    """Complex amplitude field defined bin-by-bin over the Dalitz plane.

    Each cell owns ``a_ij exp(i phi_ij)``. ``active_mask`` may be supplied to
    mark only cells intersecting the physical Dalitz region. Inactive cells are
    never intended to carry fit parameters; for linear/cubic interpolation they
    act only as ghost cells filled from the nearest active value.

    All geometry that depends only on the fixed binning/mask is cached once:
    bin edges, centres and the nearest-active ghost-cell gather map. Floating
    QMI magnitudes/phases therefore only rebuild the value field itself.
    """
    s12_edges: tuple[float,...]
    s13_edges: tuple[float,...]
    magnitudes: tuple[tuple[object,...],...]
    phases: tuple[tuple[object,...],...]
    interpolation: str = "none"
    folded: bool = False
    active_mask: tuple[tuple[bool,...],...] | None = None

    def __post_init__(self):
        if not all(isfinite(float(v)) for v in (*self.s12_edges, *self.s13_edges)):
            raise ValueError("QMI2D bin edges must be finite")
        for grid in (self.magnitudes, self.phases):
            for row in grid:
                for value in row:
                    if (isinstance(value, Parameter) and not value.fixed
                            and value.kind is not ParameterKind.DYNAMICS):
                        raise ValueError(
                            f"QMI2D node {value.name!r} must use Parameter.dynamics "
                            "with the component owner"
                        )
        if len(self.s12_edges)<2 or len(self.s13_edges)<2: raise ValueError("QMI2D requires at least one bin on each axis")
        if any(b<=a for a,b in zip(self.s12_edges[:-1],self.s12_edges[1:])): raise ValueError("QMI2D s12_edges must be strictly increasing")
        if any(b<=a for a,b in zip(self.s13_edges[:-1],self.s13_edges[1:])): raise ValueError("QMI2D s13_edges must be strictly increasing")
        nx,ny=self.shape
        if len(self.magnitudes)!=nx or any(len(r)!=ny for r in self.magnitudes): raise ValueError("QMI2D magnitudes shape must match the 2D binning")
        if len(self.phases)!=nx or any(len(r)!=ny for r in self.phases): raise ValueError("QMI2D phases shape must match the 2D binning")
        if self.active_mask is not None and (len(self.active_mask)!=nx or any(len(r)!=ny for r in self.active_mask)): raise ValueError("QMI2D active_mask shape must match the 2D binning")
        if self.interpolation not in {"none","linear","cubic"}: raise ValueError("QMI2D interpolation must be 'none', 'linear', or 'cubic'")
        if self.interpolation=="cubic" and (nx<2 or ny<2): raise ValueError("cubic QMI2D interpolation requires at least 2x2 bins")
        if self.active_mask is not None and not any(any(row) for row in self.active_mask):
            raise ValueError("QMI2D active_mask contains no physical bins")
        if self.active_mask is not None:
            for i, row in enumerate(self.active_mask):
                for j, active in enumerate(row):
                    if not active:
                        for value in (self.magnitudes[i][j], self.phases[i][j]):
                            if isinstance(value, Parameter) and not value.fixed:
                                raise ValueError(
                                    f"QMI2D inactive cell ({i}, {j}) cannot contain "
                                    f"free parameter {value.name!r}"
                                )

    @property
    def shape(self): return (len(self.s12_edges)-1,len(self.s13_edges)-1)
    @property
    def n_active_bins(self): return self.shape[0]*self.shape[1] if self.active_mask is None else sum(sum(bool(v) for v in r) for r in self.active_mask)

    @cached_property
    def _x_edges_fixed(self):
        return jnp.asarray(self.s12_edges)

    @cached_property
    def _y_edges_fixed(self):
        return jnp.asarray(self.s13_edges)

    @cached_property
    def _x_centers_fixed(self):
        return 0.5 * (self._x_edges_fixed[:-1] + self._x_edges_fixed[1:])

    @cached_property
    def _y_centers_fixed(self):
        return 0.5 * (self._y_edges_fixed[:-1] + self._y_edges_fixed[1:])

    @cached_property
    def _active_mask_fixed(self):
        return None if self.active_mask is None else jnp.asarray(self.active_mask, dtype=bool)

    @cached_property
    def _ghost_sources(self):
        return _nearest_active_sources(self.active_mask)

    def _coordinates(self,data):
        s12,s13=jnp.asarray(data["s12"]),jnp.asarray(data["s13"])
        return (jnp.minimum(s12,s13),jnp.maximum(s12,s13)) if self.folded else (s12,s13)

    def interpolated_magnitude_phase(self,data):
        x,y=self._coordinates(data)
        xe=self._x_edges_fixed.astype(x.dtype)
        ye=self._y_edges_fixed.astype(y.dtype)
        xc=self._x_centers_fixed.astype(x.dtype)
        yc=self._y_centers_fixed.astype(y.dtype)
        mag=jnp.asarray(self.magnitudes,dtype=x.dtype)
        phase=jnp.asarray(self.phases,dtype=x.dtype)
        mag=_fill_inactive_from_sources(mag,self._ghost_sources)
        phase=_fill_inactive_from_sources(phase,self._ghost_sources)
        if self.interpolation=="none":
            ix=jnp.clip(jnp.searchsorted(xe,x,side="right")-1,0,mag.shape[0]-1)
            iy=jnp.clip(jnp.searchsorted(ye,y,side="right")-1,0,mag.shape[1]-1)
            out_mag,out_phase=mag[ix,iy],phase[ix,iy]
            if self._active_mask_fixed is not None:
                active=self._active_mask_fixed[ix,iy]
                out_mag=jnp.where(active,out_mag,0.0)
                out_phase=jnp.where(active,out_phase,0.0)
            return out_mag,out_phase
        xx,yy=jnp.clip(x,xc[0],xc[-1]),jnp.clip(y,yc[0],yc[-1])
        if self.interpolation=="linear": return _bilinear(xx,yy,xc,yc,mag),_bilinear(xx,yy,xc,yc,phase)
        cubic=jax.vmap(_bicubic_one,in_axes=(0,0,None,None,None)); xf,yf=jnp.ravel(xx),jnp.ravel(yy)
        return cubic(xf,yf,xc,yc,mag).reshape(xx.shape),cubic(xf,yf,xc,yc,phase).reshape(xx.shape)

    def __call__(self,data,parameters=None):
        del parameters
        magnitude,phase=self.interpolated_magnitude_phase(data)
        return magnitude*jnp.exp(1j*phase)


__all__=["QMI2D","physical_bin_mask"]
