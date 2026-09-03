"""Two-dimensional quasi-model-independent Dalitz amplitude."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import sqrt

import jax
import jax.numpy as jnp


def _kallen(x, y, z):
    return x*x + y*y + z*z - 2.0*x*y - 2.0*x*z - 2.0*y*z


def _s13_limits_scalar(s12, mother_mass, masses):
    m1, m2, m3 = masses
    root = sqrt(max(s12, 0.0))
    e1 = (s12 + m1*m1 - m2*m2) / (2.0*root)
    e3 = (mother_mass*mother_mass - s12 - m3*m3) / (2.0*root)
    q = sqrt(max(_kallen(s12, m1*m1, m2*m2), 0.0)) / (2.0*root)
    p = sqrt(max(_kallen(mother_mass*mother_mass, s12, m3*m3), 0.0)) / (2.0*root)
    common = m1*m1 + m3*m3 + 2.0*e1*e3
    spread = 2.0*q*p
    return common-spread, common+spread


def physical_bin_mask(s12_edges, s13_edges, *, mother_mass, masses, folded=False, samples_per_bin=129):
    """Return bins that intersect the exact physical Dalitz region."""
    if samples_per_bin < 3:
        raise ValueError("samples_per_bin must be at least 3")
    xedges = tuple(float(v) for v in s12_edges)
    yedges = tuple(float(v) for v in s13_edges)
    rows = []
    for x0, x1 in zip(xedges[:-1], xedges[1:]):
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


def _bicubic_one(x, y, xc, yc, values):
    nx, ny = values.shape
    ix1, ix2, tx = _indices_and_fraction(x, xc)
    iy1, iy2, ty = _indices_and_fraction(y, yc)
    ix = jnp.clip(jnp.asarray([ix1-1,ix1,ix2,ix2+1]),0,nx-1)
    iy = jnp.clip(jnp.asarray([iy1-1,iy1,iy2,iy2+1]),0,ny-1)
    patch = values[ix[:,None],iy[None,:]]
    along = jax.vmap(lambda c: _catmull_rom(c[0],c[1],c[2],c[3],tx), in_axes=1)(patch)
    return _catmull_rom(along[0],along[1],along[2],along[3],ty)


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
