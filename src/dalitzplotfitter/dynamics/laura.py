"""Laura++-style symbolic resonance dynamics.

The functions in this module are independent of AmpForm. They construct SymPy
expressions that can be inserted into an AmpForm model through a custom dynamics
builder and subsequently compiled to JAX by TensorWaves.
"""

from __future__ import annotations

import sympy as sp


def kallen(x, y, z):
    """Return the Källén function ``lambda(x, y, z)``."""

    return x**2 + y**2 + z**2 - 2 * x * y - 2 * x * z - 2 * y * z


def breakup_momentum(mass, daughter_mass1, daughter_mass2):
    """Two-body breakup momentum in the parent rest frame.

    Parameters are masses, not squared masses. The expression is

    ``sqrt(lambda(m^2, m1^2, m2^2)) / (2 m)``.
    """

    return sp.sqrt(
        kallen(mass**2, daughter_mass1**2, daughter_mass2**2)
    ) / (2 * mass)


def _blatt_weisskopf_polynomial(z, angular_momentum: int):
    """Denominator polynomial used by the Laura++ Blatt-Weisskopf factors."""

    l = int(angular_momentum)
    if l == 0:
        return sp.Integer(1)
    if l == 1:
        return 1 + z**2
    if l == 2:
        return z**4 + 3 * z**2 + 9
    if l == 3:
        return z**6 + 6 * z**4 + 45 * z**2 + 225
    if l == 4:
        return z**8 + 10 * z**6 + 135 * z**4 + 1575 * z**2 + 11025
    if l == 5:
        return (
            z**10
            + 15 * z**8
            + 315 * z**6
            + 6300 * z**4
            + 99225 * z**2
            + 893025
        )
    raise NotImplementedError(
        "Laura++ Blatt-Weisskopf factors are currently implemented for L=0..5"
    )


def blatt_weisskopf_factor(
    mass,
    mass0,
    daughter_mass1,
    daughter_mass2,
    angular_momentum: int,
    meson_radius,
):
    """Laura++ Blatt-Weisskopf factor normalized to unity at ``mass0``.

    This implements Eqs. (15)-(20) of the Laura++ paper using ``z = q r`` and
    ``z0 = q0 r``. Consequently ``X(q0 r) = 1``.
    """

    l = int(angular_momentum)
    if l == 0:
        return sp.Integer(1)
    q = breakup_momentum(mass, daughter_mass1, daughter_mass2)
    q0 = breakup_momentum(mass0, daughter_mass1, daughter_mass2)
    z = q * meson_radius
    z0 = q0 * meson_radius
    return sp.sqrt(
        _blatt_weisskopf_polynomial(z0, l)
        / _blatt_weisskopf_polynomial(z, l)
    )


def energy_dependent_width(
    mass,
    mass0,
    gamma0,
    daughter_mass1,
    daughter_mass2,
    angular_momentum: int,
    meson_radius=None,
):
    """Laura++ mass-dependent resonance width.

    Implements Eq. (7) of Back et al., CPC 231 (2018) 198-242:

    ``Gamma(m) = Gamma0 (q/q0)^(2L+1) (m0/m) X(q r)^2``.

    If ``meson_radius`` is ``None``, the explicit Blatt-Weisskopf factor in the
    width is set to unity while retaining the threshold power of ``q``.
    """

    l = int(angular_momentum)
    q = breakup_momentum(mass, daughter_mass1, daughter_mass2)
    q0 = breakup_momentum(mass0, daughter_mass1, daughter_mass2)
    if meson_radius is None:
        barrier = sp.Integer(1)
    else:
        barrier = blatt_weisskopf_factor(
            mass,
            mass0,
            daughter_mass1,
            daughter_mass2,
            l,
            meson_radius,
        )
    return gamma0 * (q / q0) ** (2 * l + 1) * (mass0 / mass) * barrier**2


def relativistic_breit_wigner(
    mass,
    mass0,
    gamma0,
    daughter_mass1=None,
    daughter_mass2=None,
    angular_momentum: int = 0,
    meson_radius=None,
    *,
    energy_dependent: bool = True,
):
    """Laura++ relativistic Breit-Wigner mass term ``R(m)``.

    The Laura++ convention has unit numerator:

    ``R(m) = 1 / (m0^2 - m^2 - i m0 Gamma(m))``.

    This differs by an overall resonance-dependent constant from some common
    Breit-Wigner implementations. Such constants matter for the numerical scale
    of external coefficients, which is why DalitzPlotFitter defines the
    convention explicitly.
    """

    if energy_dependent:
        if daughter_mass1 is None or daughter_mass2 is None:
            raise ValueError(
                "daughter masses are required for an energy-dependent width"
            )
        width = energy_dependent_width(
            mass,
            mass0,
            gamma0,
            daughter_mass1,
            daughter_mass2,
            angular_momentum,
            meson_radius,
        )
    else:
        width = gamma0
    return 1 / (mass0**2 - mass**2 - sp.I * mass0 * width)
