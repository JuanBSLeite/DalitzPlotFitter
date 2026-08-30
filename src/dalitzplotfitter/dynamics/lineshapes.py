"""Pluggable resonance lineshapes and barrier-factor helpers.

The Gounaris-Sakurai and Flatte implementations follow the conventions collected
in Appendix A of Laura++ (J. Back et al., CPC 231 (2018) 198-242).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .context import ResonanceContext


def kallen(x, y, z):
    return x**2 + y**2 + z**2 - 2 * x * y - 2 * x * z - 2 * y * z


def breakup_momentum(mass, mass1, mass2):
    """Two-body daughter momentum in the rest frame of ``mass``."""

    mass = jnp.asarray(mass)
    radicand = jnp.maximum(kallen(mass**2, mass1**2, mass2**2), 0.0)
    return jnp.sqrt(radicand) / (2.0 * mass)


def bachelor_momentum_resonance_frame(parent_mass, resonance_mass, bachelor_mass):
    """Bachelor momentum in the resonance rest frame."""

    resonance_mass = jnp.asarray(resonance_mass)
    radicand = jnp.maximum(
        kallen(parent_mass**2, resonance_mass**2, bachelor_mass**2), 0.0
    )
    return jnp.sqrt(radicand) / (2.0 * resonance_mass)


def effective_pole_mass(context: ResonanceContext):
    """Pole mass used only for momentum calculations of virtual states."""

    m0 = jnp.asarray(context.pole_mass)
    m1, m2 = context.daughter_masses
    minimum = jnp.asarray(m1 + m2)
    maximum = jnp.asarray(context.parent_mass - context.bachelor_mass)
    span = maximum - minimum
    midpoint = 0.5 * (minimum + maximum)
    mapped = minimum + 0.5 * span * (
        1.0 + jnp.tanh((m0 - midpoint) / span)
    )
    outside = (m0 < minimum) | (m0 > maximum)
    return jnp.where(outside, mapped, m0)


def _blatt_polynomial(z, angular_momentum: int):
    l = int(angular_momentum)
    if l == 0:
        return jnp.ones_like(z)
    if l == 1:
        return 1.0 + z**2
    if l == 2:
        return z**4 + 3.0 * z**2 + 9.0
    if l == 3:
        return z**6 + 6.0 * z**4 + 45.0 * z**2 + 225.0
    if l == 4:
        return z**8 + 10.0 * z**6 + 135.0 * z**4 + 1575.0 * z**2 + 11025.0
    raise NotImplementedError("barrier factors currently support L=0..4")


def blatt_weisskopf_from_momenta(momentum, pole_momentum, angular_momentum: int, radius):
    """Blatt-Weisskopf factor normalized to unity at the pole momentum."""

    l = int(angular_momentum)
    momentum = jnp.asarray(momentum)
    if l == 0:
        return jnp.ones_like(momentum)
    z = momentum * radius
    z0 = pole_momentum * radius
    return jnp.sqrt(_blatt_polynomial(z0, l) / _blatt_polynomial(z, l))


def energy_dependent_width(mass, context: ResonanceContext):
    """Mass-dependent width used by the relativistic Breit-Wigner and GS."""

    l = int(context.spin)
    m1, m2 = context.daughter_masses
    q = breakup_momentum(mass, m1, m2)
    q0 = breakup_momentum(effective_pole_mass(context), m1, m2)
    x_res = blatt_weisskopf_from_momenta(q, q0, l, context.resonance_radius)
    safe_mass = jnp.where(mass > 0.0, mass, jnp.nan)
    safe_q0 = jnp.where(q0 > 0.0, q0, jnp.nan)
    return (
        context.pole_width
        * (q / safe_q0) ** (2 * l + 1)
        * (context.pole_mass / safe_mass)
        * x_res**2
    )


@dataclass(frozen=True)
class RelativisticBreitWigner:
    """Relativistic Breit-Wigner lineshape with running width."""

    def __call__(self, mass, context: ResonanceContext):
        width = energy_dependent_width(mass, context)
        m0 = context.pole_mass
        return 1.0 / (m0**2 - mass**2 - 1j * m0 * width)


@dataclass(frozen=True)
class GounarisSakurai:
    """Laura++ Gounaris-Sakurai lineshape for rho -> pi pi.

    The dispersive correction follows Laura++ Eqs. (38)-(43).  The pion mass
    entering h(m) is the arithmetic mean of the two daughter masses; this also
    permits the tiny charged/neutral pion-mass splitting in rho+ decays.
    """

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 1:
            raise ValueError("GounarisSakurai is defined for spin-1 rho-like states")

        m = jnp.asarray(mass)
        m0 = jnp.asarray(context.pole_mass)
        gamma0 = jnp.asarray(context.pole_width)
        mpi = 0.5 * (context.daughter_masses[0] + context.daughter_masses[1])
        q = breakup_momentum(m, mpi, mpi)
        q0 = breakup_momentum(m0, mpi, mpi)

        def h(x, qx):
            safe_q = jnp.where(qx > 0.0, qx, jnp.finfo(jnp.asarray(x).dtype).tiny)
            return (2.0 / jnp.pi) * (safe_q / x) * jnp.log((x + 2.0 * safe_q) / (2.0 * mpi))

        h_m = h(m, q)
        h_0 = h(m0, q0)
        dh_dm2_0 = h_0 * (1.0 / (8.0 * q0**2) - 1.0 / (2.0 * m0**2)) + 1.0 / (2.0 * jnp.pi * m0**2)
        f_m = gamma0 * m0**2 / q0**3 * (
            q**2 * (h_m - h_0) + (m0**2 - m**2) * q0**2 * dh_dm2_0
        )
        d = (
            (3.0 / jnp.pi) * (mpi**2 / q0**2) * jnp.log((m0 + 2.0 * q0) / (2.0 * mpi))
            + m0 / (2.0 * jnp.pi * q0)
            - mpi**2 * m0 / (jnp.pi * q0**3)
        )
        width = energy_dependent_width(m, context)
        numerator = 1.0 + d * gamma0 / m0
        return numerator / (m0**2 - m**2 + f_m - 1j * m0 * width)


def _flatte_phase_space(mass, mass_a: float, mass_b: float):
    """Laura++ Flatte phase-space factor with analytic continuation."""

    m = jnp.asarray(mass)
    threshold2 = (mass_a + mass_b) ** 2
    argument = 1.0 - threshold2 / m**2
    return jnp.sqrt(argument.astype(jnp.complex128))


@dataclass(frozen=True)
class Flatte:
    """Laura++ coupled two-channel Flatte lineshape.

    Each channel contains two charge combinations.  The first channel uses
    isospin weights (1/3, 2/3), the second (1/2, 1/2), exactly as in Laura++
    Eqs. (44)-(46). Couplings are in GeV and widths are analytically continued
    below each threshold. ``adler_zero`` is in GeV^2; ``None`` means f_A = 1.
    """

    g1: float
    g2: float
    channel1: tuple[tuple[float, float], tuple[float, float]]
    channel2: tuple[tuple[float, float], tuple[float, float]]
    adler_zero: float | None = None

    def _adler(self, mass, pole_mass):
        if self.adler_zero is None:
            return jnp.ones_like(jnp.asarray(mass))
        s_a = self.adler_zero
        return (jnp.asarray(mass) ** 2 - s_a) / (jnp.asarray(pole_mass) ** 2 - s_a)

    def widths(self, mass, context: ResonanceContext):
        f_a = self._adler(mass, context.pole_mass)
        rho11 = _flatte_phase_space(mass, *self.channel1[0])
        rho12 = _flatte_phase_space(mass, *self.channel1[1])
        rho21 = _flatte_phase_space(mass, *self.channel2[0])
        rho22 = _flatte_phase_space(mass, *self.channel2[1])
        gamma1 = self.g1 * f_a * ((1.0 / 3.0) * rho11 + (2.0 / 3.0) * rho12)
        gamma2 = self.g2 * f_a * (0.5 * rho21 + 0.5 * rho22)
        return gamma1, gamma2

    def __call__(self, mass, context: ResonanceContext):
        gamma1, gamma2 = self.widths(mass, context)
        m0 = jnp.asarray(context.pole_mass)
        m = jnp.asarray(mass)
        return 1.0 / (m0**2 - m**2 - 1j * m0 * (gamma1 + gamma2))

    @classmethod
    def f0_980(cls):
        # Laura++ Table A.2 / BES convention. Values 0.165 and 4.21*g1 are
        # quoted as m0*g for this source; divide by the nominal 0.965 GeV mass
        # to obtain the GeV couplings used in Eqs. (44)-(46).
        mpi0, mpip = 0.1349768, 0.13957039
        mkp, mk0 = 0.493677, 0.497611
        m0_ref = 0.965
        g1 = 0.165 / m0_ref
        return cls(
            g1=g1,
            g2=4.21 * g1,
            channel1=((mpi0, mpi0), (mpip, mpip)),
            channel2=((mkp, mkp), (mk0, mk0)),
        )

    @classmethod
    def k0star_1430_neutral(cls):
        mk0, mkp, mpi0, mpip, metap = 0.497611, 0.493677, 0.1349768, 0.13957039, 0.95778
        return cls(0.304, 0.380, ((mk0, mpi0), (mkp, mpip)), ((mk0, metap), (mk0, metap)), 0.234)

    @classmethod
    def k0star_1430_charged(cls):
        mk0, mkp, mpi0, mpip, metap = 0.497611, 0.493677, 0.1349768, 0.13957039, 0.95778
        return cls(0.304, 0.380, ((mkp, mpi0), (mk0, mpip)), ((mkp, metap), (mkp, metap)), 0.234)

    @classmethod
    def a0_980_neutral(cls):
        meta, mpi0, mkp, mk0 = 0.547862, 0.1349768, 0.493677, 0.497611
        m0_ref = 0.999
        g1 = 0.105 / m0_ref
        return cls(g1, 1.03 * g1, ((meta, mpi0), (meta, mpi0)), ((mkp, mkp), (mk0, mk0)))

    @classmethod
    def a0_980_charged(cls):
        meta, mpip, mkp, mk0 = 0.547862, 0.13957039, 0.493677, 0.497611
        m0_ref = 0.999
        g1 = 0.105 / m0_ref
        return cls(g1, 1.03 * g1, ((meta, mpip), (meta, mpip)), ((mkp, mk0), (mkp, mk0)))
