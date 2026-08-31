"""Square-Dalitz coordinates and deterministic integration grid."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .sample import PhaseSpaceSample


def _kallen(x, y, z):
    return x**2 + y**2 + z**2 - 2.0 * x * y - 2.0 * x * z - 2.0 * y * z


def _pair_key(i: int, j: int) -> str:
    a, b = sorted((i, j))
    if (a, b) == (0, 1):
        return "s12"
    if (a, b) == (0, 2):
        return "s13"
    if (a, b) == (1, 2):
        return "s23"
    raise ValueError("pair indices must be distinct members of {0,1,2}")


def square_dalitz_to_invariants(
    mprime,
    thetaprime,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
    pair: tuple[int, int] = (0, 1),
):
    """Map square-Dalitz coordinates to ``s12``, ``s13`` and ``s23``.

    The convention follows Laura++: for a selected two-body pair ``(i,j)``,

    ``m' = acos(2 (m_ij - m_min)/(m_max - m_min) - 1) / pi``
    and ``theta' = theta_ij / pi``.
    """

    i, j = pair
    if i == j or i not in (0, 1, 2) or j not in (0, 1, 2):
        raise ValueError("pair must contain two distinct indices from 0, 1, 2")
    k = next(index for index in range(3) if index not in pair)

    m = masses
    mi, mj, mk = m[i], m[j], m[k]
    m_min = mi + mj
    m_max = mother_mass - mk
    delta_m = m_max - m_min

    mp = jnp.asarray(mprime)
    tp = jnp.asarray(thetaprime)
    m_ij = m_min + 0.5 * delta_m * (1.0 + jnp.cos(jnp.pi * mp))
    s_ij = m_ij**2
    theta = jnp.pi * tp

    root_s = m_ij
    e_i = (s_ij + mi**2 - mj**2) / (2.0 * root_s)
    e_k = (mother_mass**2 - s_ij - mk**2) / (2.0 * root_s)
    q = jnp.sqrt(jnp.maximum(_kallen(s_ij, mi**2, mj**2), 0.0)) / (2.0 * root_s)
    p = jnp.sqrt(jnp.maximum(_kallen(mother_mass**2, s_ij, mk**2), 0.0)) / (2.0 * root_s)

    # theta is the angle between daughter i and the bachelor k in the ij frame.
    s_ik = mi**2 + mk**2 + 2.0 * (e_i * e_k - q * p * jnp.cos(theta))
    total = mother_mass**2 + sum(value**2 for value in masses)
    s_jk = total - s_ij - s_ik

    values = {
        _pair_key(i, j): s_ij,
        _pair_key(i, k): s_ik,
        _pair_key(j, k): s_jk,
    }
    return values["s12"], values["s13"], values["s23"]


def square_dalitz_jacobian(
    mprime,
    thetaprime,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
    pair: tuple[int, int] = (0, 1),
):
    """Absolute Jacobian ``|d(s_ij,s_ik)/d(m',theta')|`` for the SDP map.

    With ``m_ij = m_min + Delta_m/2 * (1 + cos(pi m'))`` and
    ``theta = pi theta'`` the determinant factorizes because ``s_ij`` is
    independent of ``theta'``::

        |J| = |ds_ij/dm'| |ds_ik/dtheta'|
            = 2 pi^2 Delta_m m_ij q p sin(pi m') sin(pi theta').
    """

    i, j = pair
    if i == j or i not in (0, 1, 2) or j not in (0, 1, 2):
        raise ValueError("pair must contain two distinct indices from 0, 1, 2")
    k = next(index for index in range(3) if index not in pair)

    mi, mj, mk = masses[i], masses[j], masses[k]
    m_min = mi + mj
    m_max = mother_mass - mk
    delta_m = m_max - m_min

    mp = jnp.asarray(mprime)
    tp = jnp.asarray(thetaprime)
    m_ij = m_min + 0.5 * delta_m * (1.0 + jnp.cos(jnp.pi * mp))
    s_ij = m_ij**2

    q = jnp.sqrt(jnp.maximum(_kallen(s_ij, mi**2, mj**2), 0.0)) / (2.0 * m_ij)
    p = jnp.sqrt(jnp.maximum(_kallen(mother_mass**2, s_ij, mk**2), 0.0)) / (2.0 * m_ij)

    ds_dmprime = jnp.pi * delta_m * m_ij * jnp.sin(jnp.pi * mp)
    ds_cross_dthetaprime = 2.0 * jnp.pi * q * p * jnp.sin(jnp.pi * tp)
    return jnp.abs(ds_dmprime * ds_cross_dthetaprime)


def invariants_to_square_dalitz(
    s12,
    s13,
    s23,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
    pair: tuple[int, int] = (0, 1),
):
    """Convert physical Dalitz invariants to Laura++ ``(m', theta')``."""

    i, j = pair
    if i == j or i not in (0, 1, 2) or j not in (0, 1, 2):
        raise ValueError("pair must contain two distinct indices from 0, 1, 2")
    k = next(index for index in range(3) if index not in pair)

    invariant = {"s12": jnp.asarray(s12), "s13": jnp.asarray(s13), "s23": jnp.asarray(s23)}
    s_ij = invariant[_pair_key(i, j)]
    s_ik = invariant[_pair_key(i, k)]

    mi, mj, mk = masses[i], masses[j], masses[k]
    m_min = mi + mj
    m_max = mother_mass - mk
    m_ij = jnp.sqrt(jnp.maximum(s_ij, 0.0))
    cosine_mass = 2.0 * (m_ij - m_min) / (m_max - m_min) - 1.0
    mprime = jnp.arccos(jnp.clip(cosine_mass, -1.0, 1.0)) / jnp.pi

    e_i = (s_ij + mi**2 - mj**2) / (2.0 * m_ij)
    e_k = (mother_mass**2 - s_ij - mk**2) / (2.0 * m_ij)
    q = jnp.sqrt(jnp.maximum(_kallen(s_ij, mi**2, mj**2), 0.0)) / (2.0 * m_ij)
    p = jnp.sqrt(jnp.maximum(_kallen(mother_mass**2, s_ij, mk**2), 0.0)) / (2.0 * m_ij)
    denom = 2.0 * q * p
    cos_theta = jnp.where(
        denom > 0.0,
        (mi**2 + mk**2 + 2.0 * e_i * e_k - s_ik) / denom,
        1.0,
    )
    thetaprime = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0)) / jnp.pi
    return mprime, thetaprime


def _quadrature_axis(n: int, quadrature: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return nodes in [0,1] and weights compatible with package ``mean`` integration.

    The integration package computes ``mean(sample.weights * f)``. Therefore
    the returned 1D stored weights are scaled by ``n`` so that the tensor
    product satisfies

        mean(W_i W_j J_ij f_ij) = sum(w_i w_j J_ij f_ij),

    where ``w_i`` are conventional quadrature weights on [0,1].
    """

    if quadrature == "midpoint":
        nodes = (np.arange(n, dtype=np.float64) + 0.5) / n
        stored_weights = np.ones(n, dtype=np.float64)
    elif quadrature in ("gauss", "gauss-legendre"):
        x, w = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (x + 1.0)
        # Conventional [0,1] weights are w/2. Multiply by n because the
        # downstream normalization matrix divides the 2D sum by n^2.
        stored_weights = 0.5 * n * w
    else:
        raise ValueError("quadrature must be 'midpoint' or 'gauss-legendre'")
    return jnp.asarray(nodes), jnp.asarray(stored_weights)


@dataclass(frozen=True)
class SquareDalitzGrid:
    """Deterministic grid in square-Dalitz coordinates with Jacobian weights.

    ``quadrature='gauss-legendre'`` is the default and uses tensor-product
    Gauss-Legendre nodes for likelihood normalization. ``quadrature='midpoint'``
    reproduces the original regular midpoint grid for diagnostics and backwards
    comparisons.
    """

    mother_mass: float
    masses: tuple[float, float, float]
    resolution: int = 800
    pair: tuple[int, int] = (0, 1)
    quadrature: str = "gauss-legendre"

    def __post_init__(self) -> None:
        if self.resolution < 2:
            raise ValueError("SquareDalitzGrid resolution must be at least 2")
        if len(self.masses) != 3:
            raise ValueError("SquareDalitzGrid requires exactly three daughter masses")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")
        i, j = self.pair
        if i == j or i not in (0, 1, 2) or j not in (0, 1, 2):
            raise ValueError("pair must contain two distinct indices from 0, 1, 2")
        if self.quadrature not in ("midpoint", "gauss", "gauss-legendre"):
            raise ValueError("quadrature must be 'midpoint' or 'gauss-legendre'")

    def sample(self) -> PhaseSpaceSample:
        n = int(self.resolution)
        axis, axis_weights = _quadrature_axis(n, self.quadrature)
        mprime, thetaprime = jnp.meshgrid(axis, axis, indexing="ij")
        w_m, w_t = jnp.meshgrid(axis_weights, axis_weights, indexing="ij")
        mp = mprime.reshape(-1)
        tp = thetaprime.reshape(-1)
        s12, s13, s23 = square_dalitz_to_invariants(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        jacobian = square_dalitz_jacobian(
            mp,
            tp,
            mother_mass=self.mother_mass,
            masses=self.masses,
            pair=self.pair,
        )
        weights = jacobian * (w_m * w_t).reshape(-1)
        return PhaseSpaceSample(s12=s12, s13=s13, s23=s23, weights=weights)


__all__ = [
    "SquareDalitzGrid",
    "invariants_to_square_dalitz",
    "square_dalitz_jacobian",
    "square_dalitz_to_invariants",
]
