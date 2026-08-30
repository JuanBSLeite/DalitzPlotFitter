"""Laura++ / Anisovich-Sarantsev pi-pi S-wave K-matrix.

This module implements the five-pole, five-channel scattering K-matrix and
P-vector production formalism collected in Appendix A of Laura++.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .context import ResonanceContext


_POLE_MASSES = jnp.asarray([0.65100, 1.20360, 1.55817, 1.21000, 1.82206])
_POLE_COUPLINGS = jnp.asarray(
    [
        [0.22889, -0.55377, 0.00000, -0.39899, -0.34639],
        [0.94128, 0.55095, 0.00000, 0.39065, 0.31503],
        [0.36856, 0.23888, 0.55639, 0.18340, 0.18681],
        [0.33650, 0.40907, 0.85679, 0.19906, -0.00984],
        [0.18171, -0.17558, -0.79658, -0.00355, 0.22358],
    ]
)
_F_SCATT_ROW = jnp.asarray([0.23399, 0.15044, -0.20545, 0.32825, 0.35412])
_S0_SCATT = -3.92637
_S0_PROD = -3.0
_S_A = 1.0
_S_A0 = -0.15

_MPI = 0.13957039
_MK = 0.493677
_META = 0.547862
_METAP = 0.95778


def _complex_value(value):
    evaluator = getattr(value, "value", None)
    if evaluator is not None:
        return jnp.asarray(evaluator(None), dtype=jnp.complex128)
    return jnp.asarray(value, dtype=jnp.complex128)


def _two_body_rho(s, mass1, mass2):
    """Laura++ two-body phase-space factor with continuation below threshold."""

    s = jnp.asarray(s)
    threshold2 = (mass1 + mass2) ** 2
    argument = 1.0 - threshold2 / s
    return jnp.sqrt(argument.astype(jnp.complex128))


def _four_pi_rho(s):
    """Anisovich-Sarantsev four-pion phase-space approximation used by Laura++."""

    s = jnp.asarray(s)
    low = (
        1.2274
        + 0.00370909 / s**2
        - 0.111203 / s
        - 6.39017 * s
        + 16.8358 * s**2
        - 21.8845 * s**3
        + 11.3153 * s**4
    )
    continuity = jnp.sqrt(1.0 - 16.0 * _MPI**2)
    high = jnp.sqrt((1.0 - 16.0 * _MPI**2 / s).astype(jnp.complex128))
    return jnp.where(s <= 1.0, continuity * low + 0.0j, high)


def _phase_space_vector(s):
    """Diagonal rho entries in Laura++ channel order.

    Channels are pi-pi, K-Kbar, 4pi, eta-eta, eta-eta'.
    """

    return jnp.stack(
        [
            _two_body_rho(s, _MPI, _MPI),
            _two_body_rho(s, _MK, _MK),
            _four_pi_rho(s),
            _two_body_rho(s, _META, _META),
            _two_body_rho(s, _META, _METAP),
        ],
        axis=-1,
    )


def _slowly_varying_factor(s, s0):
    """Laura++ factor (1 - s0/s) / (s - s0)."""

    return (1.0 - s0 / s) / (s - s0)


def _adler_factor(s):
    """Laura++ Adler-zero factor (1-sA0/s)(s-sA*m_pi^2/2)."""

    return (1.0 - _S_A0 / s) * (s - 0.5 * _S_A * _MPI**2)


def _scattering_matrix(s):
    """Five-pole Anisovich-Sarantsev K(s) with Laura++ constants."""

    s = jnp.asarray(s)
    denominators = _POLE_MASSES**2 - s[..., None]
    pole_terms = jnp.einsum(
        "...a,au,av->...uv",
        1.0 / denominators,
        _POLE_COUPLINGS,
        _POLE_COUPLINGS,
    )

    f_scatt = jnp.zeros((5, 5), dtype=s.dtype)
    f_scatt = f_scatt.at[0, :].set(_F_SCATT_ROW)
    f_scatt = f_scatt.at[:, 0].set(_F_SCATT_ROW)
    smooth = f_scatt * _slowly_varying_factor(s[..., None, None], _S0_SCATT)
    return (pole_terms + smooth) * _adler_factor(s)[..., None, None]


@dataclass(frozen=True)
class KMatrix:
    """Laura++ five-pole/five-channel pi-pi S-wave production amplitude.

    The scattering matrix is fixed to the Anisovich-Sarantsev parameters used by
    Laura++. Process-dependent production parameters are the five complex pole
    coefficients ``betas`` and five complex slowly-varying production terms
    ``f_prod``. They may be numerical complex values or ``RealImag`` objects
    containing fit ``Parameter`` instances.

    The returned scalar is channel 1 (pi-pi) of

    ``F = (I - i K rho)^(-1) P``.
    """

    betas: tuple[object, object, object, object, object] = (
        1.0 + 0.0j,
        0.0 + 0.0j,
        0.0 + 0.0j,
        0.0 + 0.0j,
        0.0 + 0.0j,
    )
    f_prod: tuple[object, object, object, object, object] = (
        0.0 + 0.0j,
        0.0 + 0.0j,
        0.0 + 0.0j,
        0.0 + 0.0j,
        0.0 + 0.0j,
    )
    s0_prod: object = _S0_PROD

    def __post_init__(self) -> None:
        if len(self.betas) != 5:
            raise ValueError("KMatrix requires five production pole coefficients")
        if len(self.f_prod) != 5:
            raise ValueError("KMatrix requires five production SVP coefficients")

    def phase_space(self, mass):
        return _phase_space_vector(jnp.asarray(mass) ** 2)

    def scattering_matrix(self, mass):
        return _scattering_matrix(jnp.asarray(mass) ** 2)

    def production_vector(self, mass):
        s = jnp.asarray(mass) ** 2
        beta = jnp.stack([_complex_value(value) for value in self.betas])
        f_prod = jnp.stack([_complex_value(value) for value in self.f_prod])
        s0_prod = jnp.asarray(self.s0_prod)

        pole_denominators = _POLE_MASSES**2 - s[..., None]
        pole = jnp.einsum(
            "a,aj,...a->...j",
            beta,
            _POLE_COUPLINGS,
            1.0 / pole_denominators,
        )
        smooth = f_prod * _slowly_varying_factor(s[..., None], s0_prod)
        return pole + smooth

    def amplitude_vector(self, mass):
        mass = jnp.asarray(mass)
        k_matrix = self.scattering_matrix(mass).astype(jnp.complex128)
        rho = self.phase_space(mass)
        production = self.production_vector(mass)

        # K rho multiplies each K column j by rho_j.
        kernel = jnp.eye(5, dtype=jnp.complex128) - 1j * k_matrix * rho[..., None, :]
        return jnp.linalg.solve(kernel, production[..., :, None])[..., 0]

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("KMatrix is defined for the scalar pi-pi S-wave")
        return self.amplitude_vector(mass)[..., 0]


__all__ = ["KMatrix"]
