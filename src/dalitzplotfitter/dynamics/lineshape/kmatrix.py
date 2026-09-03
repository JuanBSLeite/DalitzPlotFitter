"""Laura++ / Anisovich-Sarantsev pi-pi S-wave K-matrix."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..context import ResonanceContext

_POLE_MASSES = jnp.asarray([0.65100, 1.20360, 1.55817, 1.21000, 1.82206])
_POLE_COUPLINGS = jnp.asarray([
    [0.22889, -0.55377, 0.00000, -0.39899, -0.34639],
    [0.94128, 0.55095, 0.00000, 0.39065, 0.31503],
    [0.36856, 0.23888, 0.55639, 0.18340, 0.18681],
    [0.33650, 0.40907, 0.85679, 0.19906, -0.00984],
    [0.18171, -0.17558, -0.79658, -0.00355, 0.22358],
])
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
    s = jnp.asarray(s)
    argument = 1.0 - (mass1 + mass2) ** 2 / s
    return jnp.sqrt(argument.astype(jnp.complex128))


def _four_pi_rho(s):
    s = jnp.asarray(s)
    low = 1.2274 + 0.00370909 / s**2 - 0.111203 / s - 6.39017 * s + 16.8358 * s**2 - 21.8845 * s**3 + 11.3153 * s**4
    continuity = jnp.sqrt(1.0 - 16.0 * _MPI**2)
    high = jnp.sqrt((1.0 - 16.0 * _MPI**2 / s).astype(jnp.complex128))
    return jnp.where(s <= 1.0, continuity * low + 0.0j, high)


def _phase_space_vector(s):
    return jnp.stack([
        _two_body_rho(s, _MPI, _MPI),
        _two_body_rho(s, _MK, _MK),
        _four_pi_rho(s),
        _two_body_rho(s, _META, _META),
        _two_body_rho(s, _META, _METAP),
    ], axis=-1)


def _slowly_varying_factor(s, s0):
    return (1.0 - s0 / s) / (s - s0)


def _adler_factor(s):
    return (1.0 - _S_A0 / s) * (s - 0.5 * _S_A * _MPI**2)


def _stable_inverse_denominators(s):
    denominators = _POLE_MASSES**2 - s[..., None]
    eps = jnp.finfo(s.dtype).eps
    scale = jnp.maximum(1.0, jnp.abs(_POLE_MASSES**2))
    regularized = jnp.where(
        jnp.abs(denominators) <= eps * scale,
        jnp.where(denominators < 0.0, -eps * scale, eps * scale),
        denominators,
    )
    return 1.0 / regularized


def _scattering_matrix_from_inverse(s, inverse_denominators):
    pole_terms = jnp.einsum(
        "...a,au,av->...uv",
        inverse_denominators,
        _POLE_COUPLINGS,
        _POLE_COUPLINGS,
    )
    f_scatt = jnp.zeros((5, 5), dtype=s.dtype)
    f_scatt = f_scatt.at[0, :].set(_F_SCATT_ROW)
    f_scatt = f_scatt.at[:, 0].set(_F_SCATT_ROW)
    smooth = f_scatt * _slowly_varying_factor(
        s[..., None, None], _S0_SCATT
    )
    return (pole_terms + smooth) * _adler_factor(s)[..., None, None]


def _scattering_matrix(s):
    s = jnp.asarray(s)
    return _scattering_matrix_from_inverse(s, _stable_inverse_denominators(s))


def _kernel(mass):
    mass = jnp.asarray(mass)
    s = mass**2
    inverse_denominators = _stable_inverse_denominators(s)
    k_matrix = _scattering_matrix_from_inverse(
        s, inverse_denominators
    ).astype(jnp.complex128)
    rho = _phase_space_vector(s)
    kernel = jnp.eye(5, dtype=jnp.complex128) - 1j * k_matrix * rho[..., None, :]
    return s, inverse_denominators, k_matrix, rho, kernel


@dataclass(frozen=True)
class KMatrix:
    betas: tuple[object, object, object, object, object] = (1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j)
    f_prod: tuple[object, object, object, object, object] = (0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j)
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

    def scattering_amplitude(self, mass):
        _, _, k_matrix, _, kernel = _kernel(mass)
        return jnp.linalg.solve(kernel, k_matrix)

    def s_matrix(self, mass):
        mass = jnp.asarray(mass)
        rho = self.phase_space(mass)
        sqrt_rho = jnp.sqrt(rho)
        t_matrix = self.scattering_amplitude(mass)
        dressed = sqrt_rho[..., :, None] * t_matrix * sqrt_rho[..., None, :]
        return jnp.eye(5, dtype=jnp.complex128) + 2j * dressed

    def _production_vector_from_inverse(self, s, inverse_denominators):
        beta = jnp.stack([_complex_value(value) for value in self.betas])
        f_prod = jnp.stack([_complex_value(value) for value in self.f_prod])
        s0_prod = jnp.asarray(self.s0_prod)
        pole = jnp.einsum(
            "a,aj,...a->...j",
            beta,
            _POLE_COUPLINGS,
            inverse_denominators,
        )
        smooth = f_prod * _slowly_varying_factor(s[..., None], s0_prod)
        return pole + smooth

    def production_vector(self, mass):
        s = jnp.asarray(mass) ** 2
        inverse_denominators = _stable_inverse_denominators(s)
        return self._production_vector_from_inverse(s, inverse_denominators)

    def prepare_mass(self, mass, context=None):
        """Prepare the fixed pi-pi response needed by the observed channel.

        Only row zero of ``(I - i K rho)^-1`` is retained because ``__call__``
        returns the pi-pi production channel. This replaces a 5x5 solve at every
        likelihood evaluation with a five-term complex dot product, while using
        one fifth of the memory of a full inverse response matrix.
        """

        del context
        _, _, _, _, kernel = _kernel(mass)
        rhs = jnp.zeros(kernel.shape[:-1] + (1,), dtype=jnp.complex128)
        rhs = rhs.at[..., 0, 0].set(1.0 + 0.0j)
        response_column = jnp.linalg.solve(
            jnp.swapaxes(kernel, -1, -2), rhs
        )[..., 0]
        return response_column

    def evaluate_prepared(self, mass, response_row, context=None):
        """Evaluate the pi-pi production amplitude from a prepared response."""

        del context
        production = self.production_vector(mass)
        return jnp.einsum("...j,...j->...", response_row, production)

    def amplitude_vector(self, mass):
        mass = jnp.asarray(mass)
        s, inverse_denominators, _, _, kernel = _kernel(mass)
        production = self._production_vector_from_inverse(s, inverse_denominators)
        return jnp.linalg.solve(kernel, production[..., :, None])[..., 0]

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("KMatrix is defined for the scalar pi-pi S-wave")
        response_row = self.prepare_mass(mass, context)
        return self.evaluate_prepared(mass, response_row, context)


__all__ = ["KMatrix"]
