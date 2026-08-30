"""Coupled-channel Flatte lineshape."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..context import ResonanceContext


def _phase_space(mass, mass_a: float, mass_b: float):
    m = jnp.asarray(mass)
    threshold2 = (mass_a + mass_b) ** 2
    argument = 1.0 - threshold2 / m**2
    return jnp.sqrt(argument.astype(jnp.complex128))


@dataclass(frozen=True)
class Flatte:
    """Laura++ coupled two-channel Flatte lineshape."""

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
        rho11 = _phase_space(mass, *self.channel1[0])
        rho12 = _phase_space(mass, *self.channel1[1])
        rho21 = _phase_space(mass, *self.channel2[0])
        rho22 = _phase_space(mass, *self.channel2[1])
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
        mpi0, mpip = 0.1349768, 0.13957039
        mkp, mk0 = 0.493677, 0.497611
        m0_ref = 0.965
        g1 = 0.165 / m0_ref
        return cls(g1=g1, g2=4.21 * g1, channel1=((mpi0, mpi0), (mpip, mpip)), channel2=((mkp, mkp), (mk0, mk0)))

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


__all__ = ["Flatte"]
