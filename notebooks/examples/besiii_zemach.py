"""Covariant Zemach (Rarita-Schwinger) spin factors of arXiv:2312.02524, Eqs. 6-10, Table 5.

Everything is built from the four-momenta in the D0 frame (metric +---); all
expressions are Lorentz scalars. Contravariant components are stored, and
``lower`` applies the metric where a covariant index is needed.

    U = S * B_LD(q_D) * P_R1 * B_LR1(q_1) * P_R2 * B_LR2(q_2)          (Eq. 6)

with the barrier factors exactly as printed in Eqs. 11-13 (they keep the
constants sqrt(2) R, sqrt(13) R^2 that the package's unit-normalised 1/sqrt(P_L)
drops) and the tensors t^(L) = (-1)^L P^(L)(p_a) r...r, r = p_b - p_c (Eq. 7).

The Levi-Civita symbol uses eps^{0123} = +1 (an overall sign per wave is a
convention absorbed in the coupling phase).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import jax.numpy as jnp
import numpy as np

from jaxpwa.dynamics.context import ResonanceContext
from jaxpwa.dynamics.lineshape.common import breakup_momentum
from jaxpwa.dynamics.sequential import Isobar
from jaxpwa.kinematics.vectors import invariant_mass_squared

METRIC = jnp.array([1.0, -1.0, -1.0, -1.0])
_G = jnp.diag(METRIC)


def _levi_civita():
    eps = np.zeros((4, 4, 4, 4))
    for perm in permutations(range(4)):
        sign = np.linalg.det(np.eye(4)[list(perm)])
        eps[perm] = round(sign)
    return jnp.asarray(eps)


EPS = _levi_civita()


def dot(a, b):
    return jnp.sum(a * b * METRIC, axis=-1)


def lower(v):
    return v * METRIC


def lower2(t):
    return t * METRIC[:, None] * METRIC[None, :]


def proj1(p):
    """P^(1)mu nu = -g^{mu nu} + p^mu p^nu / p^2 (Eq. 9)."""
    return -_G[None] + p[:, :, None] * p[:, None, :] / dot(p, p)[:, None, None]


def t1(p, r):
    """t^(1) = -P^(1) r, transverse to p: r - p (p.r)/p^2."""
    return r - p * (dot(p, r) / dot(p, p))[:, None]


def t2(p, r):
    """t^(2)mu nu = P^(2) r r = t1 t1 + (t1.t1/3) P^(1)  (Eqs. 7, 10)."""
    t = t1(p, r)
    return t[:, :, None] * t[:, None, :] + (dot(t, t) / 3.0)[:, None, None] * proj1(p)


def barrier(q, orbital, radius):
    """Blatt-Weisskopf factors exactly as printed in Eqs. 11-13 (q_R = 1/R)."""
    q_r = 1.0 / radius
    if orbital == 0:
        return jnp.ones_like(q)
    if orbital == 1:
        return jnp.sqrt(2.0 / (q**2 + q_r**2))
    if orbital == 2:
        return jnp.sqrt(13.0 / (q**4 + 3.0 * q**2 * q_r**2 + 9.0 * q_r**4))
    raise NotImplementedError("orbital 0..2 only (Eqs. 11-13)")


def _mass(p):
    return jnp.sqrt(jnp.maximum(invariant_mass_squared(p), 0.0))


def propagator(iso: Isobar, mass, m1, m2, parent_mass, bachelor_mass, orbital, values=None):
    """Isobar lineshape alone (no q^L: the tensors carry the momentum factors)."""
    from jaxpwa.dynamics.context import resolve_value

    context = ResonanceContext(
        parent_mass=parent_mass,
        daughter_masses=(m1, m2),
        bachelor_mass=bachelor_mass,
        spin=orbital,
        pole_mass=resolve_value(iso.mass, values),
        pole_width=resolve_value(iso.width, values),
        resonance_radius=resolve_value(iso.radius, values),
    )
    return resolve_value(iso.lineshape, values)(mass, context)


def _vertex(iso, mass, m1, m2, parent_mass, bachelor_mass, orbital, values):
    q = breakup_momentum(mass, m1, m2)
    return propagator(iso, mass, m1, m2, parent_mass, bachelor_mass, orbital, values) * barrier(
        q, orbital, iso.radius
    )


@dataclass(frozen=True, eq=False)
class ZCascade:
    """D -> R a, R -> V b, V -> c d  (Table 5 rows with a pseudoscalar bachelor).

    ``kind``: pi_rho, pi_S (R = pi(1300)); a_rhoS, a_rhoD, a_fP, a_SP (R = a1);
    a2_rhoD (R = a2). ``l_d`` is the D orbital momentum, ``l_r`` that of R -> V b.
    """

    kind: str
    outer: Isobar
    inner: Isobar
    l_d: int
    l_r: int
    order: tuple = (0, 1, 2, 3)
    radius_d: float = 5.0

    def spin_factor(self, pa, pb, pc, pd):
        pv, = (pc + pd,)
        pr = pb + pv
        pdd = pa + pr
        r_d, r_r, r_v = pr - pa, pv - pb, pc - pd
        if self.kind == "pi_rho":
            return dot(t1(pr, r_r), t1(pv, r_v))
        if self.kind == "pi_S":
            return jnp.ones(pa.shape[0])
        td = t1(pdd, r_d)
        if self.kind == "a_rhoS":
            return jnp.einsum("nm,nmv,nv->n", lower(td), proj1(pr), lower(t1(pv, r_v)))
        if self.kind == "a_rhoD":
            return jnp.einsum("nm,nmv,nv->n", lower(td), t2(pr, r_r), lower(t1(pv, r_v)))
        if self.kind == "a_SP":
            return dot(td, t1(pr, r_r))
        if self.kind == "a_fP":
            x = lower(td)[:, :, None] * lower(t1(pr, r_r))[:, None, :]
            y = lower2(t2(pv, r_v))
            p = proj1(pr)
            t_a = jnp.einsum("nmv,nmr,nvs,nrs->n", x, p, p, y)
            t_b = jnp.einsum("nmv,nms,nvr,nrs->n", x, p, p, y)
            return 0.5 * (t_a + t_b) - jnp.einsum("nmv,nmv->n", p, x) * jnp.einsum(
                "nmv,nmv->n", p, y
            ) / 3.0
        if self.kind == "a2_rhoD":
            td2 = lower2(t2(pdd, pr - pa))
            p_t = proj1(pr)
            y = jnp.einsum("nam,nmv,nvb->nab", p_t, td2, p_t) - p_t * (
                jnp.einsum("nmv,nmv->n", p_t, td2) / 3.0
            )[:, None, None]
            w = jnp.einsum("nab,nlb->nal", lower2(y), t2(pr, r_r)) * METRIC[None, None, :]
            v_up = jnp.einsum("nsg,ng->ns", p_t, lower(t1(pv, r_v)))
            return jnp.einsum("alsr,nal,ns,nr->n", EPS, w, lower(v_up), lower(pr))
        raise KeyError(self.kind)

    def __call__(self, data, parameters=None):
        p = data["momenta"][:, list(self.order), :]
        pa, pb, pc, pd = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        pv = pc + pd
        pr = pb + pv
        m_a, m_b, m_c, m_d = (_mass(x) for x in (pa, pb, pc, pd))
        m_v, m_r, m_d0 = _mass(pv), _mass(pr), _mass(pr + pa)
        outer = _vertex(self.outer, m_r, m_v, m_b, m_d0, m_a, self.l_r, parameters)
        inner = _vertex(self.inner, m_v, m_c, m_d, m_r, m_b, self.inner.spin, parameters)
        q_d = breakup_momentum(m_d0, m_r, m_a)
        return (
            self.spin_factor(pa, pb, pc, pd)
            * barrier(q_d, self.l_d, self.radius_d)
            * outer
            * inner
        )


@dataclass(frozen=True, eq=False)
class ZPair:
    """D -> R1 R2, R1 -> a b, R2 -> c d.

    ``kind``: VV_S, VV_P, VV_D (two vectors); VS_P (vector + scalar);
    TS_D (tensor + scalar). ``l_d`` is the D orbital momentum.
    """

    kind: str
    first: Isobar
    second: Isobar
    l_d: int
    order: tuple = (0, 1, 2, 3)
    radius_d: float = 5.0

    def spin_factor(self, pa, pb, pc, pd):
        p1, p2 = pa + pb, pc + pd
        pdd = p1 + p2
        r_d, r1, r2 = p1 - p2, pa - pb, pc - pd
        if self.kind == "VV_S":
            return dot(t1(p1, r1), t1(p2, r2))
        if self.kind == "VV_P":
            return jnp.linalg.det(jnp.stack([t1(pdd, r_d), t1(p1, r1), t1(p2, r2), pdd], axis=1))
        if self.kind == "VV_D":
            return jnp.einsum("nmv,nm,nv->n", lower2(t2(pdd, r_d)), t1(p1, r1), t1(p2, r2))
        if self.kind == "VS_P":
            return dot(t1(pdd, r_d), t1(p1, r1))
        if self.kind == "TS_D":
            return jnp.einsum("nmv,nmv->n", lower2(t2(pdd, r_d)), t2(p1, r1))
        raise KeyError(self.kind)

    def __call__(self, data, parameters=None):
        p = data["momenta"][:, list(self.order), :]
        pa, pb, pc, pd = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        p1, p2 = pa + pb, pc + pd
        m = [_mass(x) for x in (pa, pb, pc, pd)]
        m1, m2, m_d0 = _mass(p1), _mass(p2), _mass(p1 + p2)
        first = _vertex(self.first, m1, m[0], m[1], m_d0, m2, self.first.spin, parameters)
        second = _vertex(self.second, m2, m[2], m[3], m_d0, m1, self.second.spin, parameters)
        q_d = breakup_momentum(m_d0, m1, m2)
        return (
            self.spin_factor(pa, pb, pc, pd)
            * barrier(q_d, self.l_d, self.radius_d)
            * first
            * second
        )
