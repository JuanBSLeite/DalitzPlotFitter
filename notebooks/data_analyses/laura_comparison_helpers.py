"""Adapters for the histogram conventions in GenFit3piCP_R2_415489.cc.

These are analysis-specific: the package's ordinary Square-Dalitz histogram
classes return bin values and do not assume those values are a density in SDP.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from dalitzplotfitter import (
    SquareDalitzHistogramBackground,
    SquareDalitzHistogramEfficiency,
    square_dalitz_jacobian,
)


def interpolated_histogram(histogram, data):
    """Laura++ bilinear interpolation, constant beyond outer bin centres.

    Matches Lau2DHistDP/Lau2DHistDPPdf::interpolateXY on rectangular SDP
    histograms. Vetoes are applied separately, after interpolation.
    """
    mp, tp = histogram.square_coordinates(data)
    xc = (histogram.mprime_edges[:-1] + histogram.mprime_edges[1:]) / 2
    yc = (histogram.thetaprime_edges[:-1] + histogram.thetaprime_edges[1:]) / 2
    x = jnp.clip(mp, xc[0], xc[-1])
    y = jnp.clip(tp, yc[0], yc[-1])
    ix = jnp.clip(jnp.searchsorted(xc, x) - 1, 0, max(len(xc) - 2, 0))
    iy = jnp.clip(jnp.searchsorted(yc, y) - 1, 0, max(len(yc) - 2, 0))
    jx = jnp.minimum(ix + 1, len(xc) - 1)
    jy = jnp.minimum(iy + 1, len(yc) - 1)
    tx = (x - xc[ix]) / jnp.where(jx != ix, xc[jx] - xc[ix], 1.0)
    ty = (y - yc[iy]) / jnp.where(jy != iy, yc[jy] - yc[iy], 1.0)
    v = histogram.values
    result = (
        (1 - tx) * (1 - ty) * v[ix, iy]
        + tx * (1 - ty) * v[jx, iy]
        + (1 - tx) * ty * v[ix, jy]
        + tx * ty * v[jx, jy]
    )
    valid = (
        (mp >= histogram.mprime_edges[0])
        & (mp <= histogram.mprime_edges[-1])
        & (tp >= histogram.thetaprime_edges[0])
        & (tp <= histogram.thetaprime_edges[-1])
    )
    return jnp.where(valid, result, 0.0)


@dataclass(frozen=True)
class LauraEfficiency(SquareDalitzHistogramEfficiency):
    """Dimensionless interpolated ACC; no coordinate Jacobian is applied."""

    def __call__(self, data):
        return interpolated_histogram(self, data)


@dataclass(frozen=True)
class LauraBackground(SquareDalitzHistogramBackground):
    """SDP density converted to ds13 ds23, as LauBkgndDPModel does.

    Folding does not insert a factor two here: normalization integrates the
    unfolded plane and therefore accounts for both identical-particle halves.
    """

    def __call__(self, data):
        mp, tp = self.square_coordinates(data)
        jacobian = square_dalitz_jacobian(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        value = interpolated_histogram(self, data)
        return jnp.where(
            jacobian > 0.0,
            value / jnp.where(jacobian > 0.0, jacobian, 1.0),
            0.0,
        )


def charge_scaled_background(shape, sample, veto, asymmetry, charge):
    """Normalize after veto, then encode the fixed counting asymmetry.

    CPBackgroundCategory splits a yield by the integrals of its two shapes.
    Using the same quadrature as CPFitSession makes those integrals exactly
    1 - charge * asymmetry, including vetoes and numerical integration error.
    """
    if charge not in (-1, 1):
        raise ValueError("charge must be +1 or -1")
    data = sample.as_dict()
    acceptance = 1.0 if veto is None else veto(data)
    normalization = float(jnp.mean(sample.weights * acceptance * shape(data)))
    if not normalization > 0.0:
        raise ValueError("background must have a positive post-veto integral")
    scale = (1.0 - charge * asymmetry) / normalization
    return lambda data: scale * shape(data)
