"""Shared interpolation kernels for two-dimensional histogram models."""

from __future__ import annotations

import jax.numpy as jnp
import jax


def interpolate_2d(x, y, x_edges, y_edges, values, mode: str):
    """Evaluate a histogram, optionally bilinearly between bin centres.

    ``mode='none'`` is the ordinary piecewise-constant lookup. ``mode='linear'``
    matches Laura++'s ``Lau2DHistDP::interpolateXY`` on rectangular bins:
    values are attached to bin centres and clamped to the outer centres.
    Points outside the histogram edges return zero.
    """
    if mode == "none":
        ix = jnp.searchsorted(x_edges, x, side="right") - 1
        iy = jnp.searchsorted(y_edges, y, side="right") - 1
        safe_ix = jnp.clip(ix, 0, values.shape[0] - 1)
        safe_iy = jnp.clip(iy, 0, values.shape[1] - 1)
        result = values[safe_ix, safe_iy]
    elif mode in {"linear", "spline"}:
        xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        yc = 0.5 * (y_edges[:-1] + y_edges[1:])
        xx = jnp.clip(x, xc[0], xc[-1])
        yy = jnp.clip(y, yc[0], yc[-1])
        ix = jnp.clip(jnp.searchsorted(xc, xx, side="right") - 1, 0, len(xc) - 2)
        iy = jnp.clip(jnp.searchsorted(yc, yy, side="right") - 1, 0, len(yc) - 2)
        jx, jy = ix + 1, iy + 1
        tx = (xx - xc[ix]) / (xc[jx] - xc[ix])
        ty = (yy - yc[iy]) / (yc[jy] - yc[iy])
        result = ((1 - tx) * (1 - ty) * values[ix, iy]
                  + tx * (1 - ty) * values[jx, iy]
                  + (1 - tx) * ty * values[ix, jy]
                  + tx * ty * values[jx, jy])
        if mode == "spline":
            def cubic_axis(v, coords, fraction):
                p0, p1, p2, p3 = v
                t = fraction
                # Catmull-Rom/Hermite cubic with repeated edge support.
                m1 = 0.5 * (p2 - p0)
                m2 = 0.5 * (p3 - p1)
                return ((2*t**3 - 3*t**2 + 1)*p1
                        + (t**3 - 2*t**2 + t)*m1
                        + (-2*t**3 + 3*t**2)*p2
                        + (t**3 - t**2)*m2)
            ix0 = jnp.clip(ix - 1, 0, values.shape[0] - 1)
            ix3 = jnp.clip(jx + 1, 0, values.shape[0] - 1)
            iy0 = jnp.clip(iy - 1, 0, values.shape[1] - 1)
            iy3 = jnp.clip(jy + 1, 0, values.shape[1] - 1)
            patch = values[jnp.stack((ix0, ix, jx, ix3))[:, None],
                           jnp.stack((iy0, iy, jy, iy3))[None, :]]
            rows = jax.vmap(lambda row: cubic_axis(row, None, ty))(patch)
            result = cubic_axis(rows, None, tx)
    else:
        raise ValueError("interpolation must be 'none', 'linear' or 'spline'")
    valid = ((x >= x_edges[0]) & (x <= x_edges[-1])
             & (y >= y_edges[0]) & (y <= y_edges[-1]))
    return jnp.where(valid, result, 0.0)
