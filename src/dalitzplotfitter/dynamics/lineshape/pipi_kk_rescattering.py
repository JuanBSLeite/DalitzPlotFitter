"""The pi-pi <-> K-Kbar rescattering shape used in the LHCb isobar model."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..context import ResonanceContext


@dataclass(frozen=True)
class PipiKKRescattering:
    """LHCb's phenomenological pi-pi to K-Kbar S-wave amplitude.

    The shape follows Eqs. (17)--(21) of Phys. Rev. D 101, 012006.  It is
    defined only in the 1.0--1.5 GeV mass window used by the isobar fit and is
    zero outside that window.

    ``convention='laura'`` uses the source factor in ``m**2`` and the
    overall phase ``i`` of Laura++ 3.8 ``LauRescatteringRes::amplitude``.
    The default ``'paper'`` retains the literal printed Eqs. (17)--(21).
    In either convention the explicit mass window is retained; Laura++ itself
    starts at the KK threshold, so set mass_min accordingly to reproduce it.
    The delta_*_squared parameters denote the source denominators: they have
    units GeV for 'paper' and GeV**2 (lambda**2) for 'laura'.
    """

    mass_min: float = 1.0
    mass_max: float = 1.5
    delta_pipi_squared: float = 1.0
    delta_kk_squared: float = 1.0
    kaon_mass: float = 0.494
    epsilon1: float = 2.4
    epsilon2: float = -5.5
    m_prime: float = 1.5
    m_f: float = 1.32
    m_s: float = 0.92
    c0: float = 1.3
    convention: str = "paper"

    def __post_init__(self) -> None:
        if self.convention not in {"paper", "laura"}:
            raise ValueError("rescattering convention must be 'paper' or 'laura'")
        if self.mass_min >= self.mass_max:
            raise ValueError("PipiKKRescattering requires mass_min < mass_max")
        if self.delta_pipi_squared <= 0.0 or self.delta_kk_squared <= 0.0:
            raise ValueError("rescattering source scales must be positive")

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("PipiKKRescattering is defined for a spin-0 S-wave")

        m = jnp.asarray(mass)
        tiny = jnp.finfo(m.dtype).tiny
        s = m**2
        s_safe = jnp.maximum(s, tiny)
        m_safe = jnp.maximum(m, tiny)
        k2_squared = jnp.maximum(0.25 * s - self.kaon_mass**2, 0.0)
        k2 = jnp.sqrt(k2_squared)
        k2_safe = jnp.maximum(k2, tiny)

        eta = 1.0 - (
            self.epsilon1 * k2 / m_safe
            + self.epsilon2 * k2**2 / s_safe
        ) * (self.m_prime**2 - s_safe) / s_safe
        cot_delta = self.c0 * (
            (s_safe - self.m_s**2) * (self.m_f**2 - s_safe)
            / (self.m_f**2 * m_safe)
        ) * jnp.abs(k2) / k2_safe**2
        exp_2i_delta = (cot_delta + 1j) / (cot_delta - 1j)
        scattering = jnp.sqrt(jnp.maximum(1.0 - eta**2, 0.0)) * exp_2i_delta
        source_variable = s if self.convention == "laura" else m
        source = 1.0 / (1.0 + source_variable / self.delta_pipi_squared) / (
            1.0 + source_variable / self.delta_kk_squared
        )
        value = source * scattering
        if self.convention == "laura":
            value = 1j * value
        inside = (m >= self.mass_min) & (m <= self.mass_max)
        return jnp.where(inside, value, 0.0j)


__all__ = ["PipiKKRescattering"]
