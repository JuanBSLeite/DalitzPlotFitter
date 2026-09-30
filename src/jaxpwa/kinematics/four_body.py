"""Five-dimensional four-body coordinates with signed helicity azimuths."""

from __future__ import annotations

import jax.numpy as jnp

from .covariant import boost_to_rest_frame
from .phase_space_mc import _boost_from_rest
from .vectors import invariant_mass_squared


def _unit(v):
    norm = jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True))
    return v / jnp.where(norm > 0, norm, 1)


def _cos(a, b):
    return jnp.clip(jnp.sum(_unit(a) * _unit(b), axis=-1), -1, 1)


def _azimuth(y, x):
    # Undefined boundary azimuths are assigned zero; boundaries have zero measure.
    valid = x * x + y * y > 0
    return jnp.arctan2(jnp.where(valid, y, 0), jnp.where(valid, x, 1))


def _ordered(momenta, order):
    if tuple(sorted(order)) != (0, 1, 2, 3):
        raise ValueError("order must be a permutation of (0, 1, 2, 3)")
    p = jnp.asarray(momenta)
    if p.ndim != 3 or p.shape[1:] != (4, 4):
        raise ValueError("four-body momenta must have shape (N, 4, 4)")
    return tuple(p[:, i] for i in order)


def pair_coordinates(momenta, order=(0, 1, 2, 3)):
    """Coordinates for P -> (a b)(c d), in the supplied particle order.

    theta1/2 use each resonance's flight direction in P. Fix a's azimuth to
    zero around R1's +z; R2 uses axes (-x, y, -z), i.e. R_y(pi). ``phi`` is
    c's azimuth in these R2 axes. The angular amplitude therefore contains
    exp(+i*lambda*phi). These coordinates are invariant under proper Lorentz
    transformations; parity reverses the signed azimuth.
    """
    a, b, c, d = _ordered(momenta, order)
    r1, r2 = a + b, c + d
    parent = r1 + r2
    a_p = boost_to_rest_frame(a, parent)
    c_p = boost_to_rest_frame(c, parent)
    r1_p = boost_to_rest_frame(r1, parent)
    r2_p = boost_to_rest_frame(r2, parent)
    a_r = boost_to_rest_frame(a_p, r1_p)[..., 1:]
    c_r = boost_to_rest_frame(c_p, r2_p)[..., 1:]
    z = _unit(r1_p[..., 1:])
    x = _unit(a_r - jnp.sum(a_r * z, axis=-1, keepdims=True) * z)
    y = jnp.cross(z, x)
    return {
        "s_ab": invariant_mass_squared(r1),
        "s_cd": invariant_mass_squared(r2),
        "cos_theta1": _cos(a_r, z),
        "cos_theta2": _cos(c_r, -z),
        "phi": _azimuth(jnp.sum(c_r * y, axis=-1), -jnp.sum(c_r * x, axis=-1)),
    }


def cascade_coordinates(momenta, order=(0, 1, 2, 3)):
    """Coordinates for P -> R a, R -> S b, S -> c d.

    theta_R is S's polar angle about -a in R. Boosting along S preserves the
    helicity axes (x,y,z)=(y cross z, z_R cross z, S/|S|); theta_S and phi
    describe c in those axes. Sequential boosts, not independently oriented
    lab-frame rest systems, keep the relative helicity phase consistent.
    """
    a, b, c, d = _ordered(momenta, order)
    s, r = c + d, b + c + d
    a_r = boost_to_rest_frame(a, r)
    s_r = boost_to_rest_frame(s, r)
    c_r = boost_to_rest_frame(c, r)
    c_s = boost_to_rest_frame(c_r, s_r)[..., 1:]
    z_r = -_unit(a_r[..., 1:])
    z = _unit(s_r[..., 1:])
    y = _unit(jnp.cross(z_r, z))
    x = jnp.cross(y, z)
    return {
        "s_bcd": invariant_mass_squared(r),
        "s_cd": invariant_mass_squared(s),
        "cos_theta_R": _cos(z_r, z),
        "cos_theta_S": _cos(c_s, z),
        "phi": _azimuth(jnp.sum(c_s * y, axis=-1), jnp.sum(c_s * x, axis=-1)),
    }


def pair_coordinates_to_momenta(
    parent_mass, masses, s_ab, s_cd, cos_theta1, cos_theta2, phi
):
    """Reconstruct canonical rest-frame momenta from five invariant coordinates.

    R1 flies along +z and a lies in the xz plane with positive x. The omitted
    global orientation is irrelevant for a scalar parent. Broadcast scalar or
    vector inputs; the output has shape ``(..., 4, 4)``. Unphysical coordinates
    produce NaNs. Threshold angles are conventional and not invertible.
    """
    if len(masses) != 4:
        raise ValueError("four daughter masses are required")
    s1, s2, c1, c2, phi = jnp.broadcast_arrays(s_ab, s_cd, cos_theta1, cos_theta2, phi)
    ma, mb, mc, md = masses
    m1, m2 = jnp.sqrt(jnp.maximum(s1, 0)), jnp.sqrt(jnp.maximum(s2, 0))

    def momentum(mother, a, b):
        q2 = (mother - a - b) * (mother + a + b) * (mother - a + b) * (mother + a - b)
        return jnp.sqrt(jnp.maximum(q2, 0)) / (2 * jnp.where(mother > 0, mother, 1))

    q, q1, q2 = (
        momentum(parent_mass, m1, m2),
        momentum(m1, ma, mb),
        momentum(m2, mc, md),
    )
    zero = jnp.zeros_like(q)
    r1_beta = jnp.stack((zero, zero, q / jnp.sqrt(s1 + q * q)), axis=-1)
    r2_beta = jnp.stack((zero, zero, -q / jnp.sqrt(s2 + q * q)), axis=-1)
    t1, t2 = (
        jnp.sqrt(jnp.maximum(1 - c1 * c1, 0)),
        jnp.sqrt(jnp.maximum(1 - c2 * c2, 0)),
    )
    a_vec = q1[..., None] * jnp.stack((t1, zero, c1), axis=-1)
    c_vec = q2[..., None] * jnp.stack(
        (-t2 * jnp.cos(phi), t2 * jnp.sin(phi), -c2), axis=-1
    )
    result = jnp.stack(
        (
            _boost_from_rest(jnp.sqrt(ma * ma + q1 * q1), a_vec, r1_beta),
            _boost_from_rest(jnp.sqrt(mb * mb + q1 * q1), -a_vec, r1_beta),
            _boost_from_rest(jnp.sqrt(mc * mc + q2 * q2), c_vec, r2_beta),
            _boost_from_rest(jnp.sqrt(md * md + q2 * q2), -c_vec, r2_beta),
        ),
        axis=-2,
    )
    valid = (
        (s1 > 0)
        & (s2 > 0)
        & (m1 >= ma + mb)
        & (m2 >= mc + md)
        & (m1 + m2 <= parent_mass)
        & (jnp.abs(c1) <= 1)
        & (jnp.abs(c2) <= 1)
        & jnp.isfinite(phi)
    )
    return jnp.where(valid[..., None, None], result, jnp.nan)
