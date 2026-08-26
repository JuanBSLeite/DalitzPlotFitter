"""Complex coefficient parameterisations from Laura++ Table 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from .base import Flavor

Scalar = float | Array


def _phase(angle: Scalar) -> Array:
    return jnp.exp(1j * jnp.asarray(angle))


def _resolve(value: object, values: Mapping[str, object] | None = None):
    resolver = getattr(value, "resolve", None)
    if resolver is not None:
        return resolver(values)
    return value


@dataclass(frozen=True)
class MagPhase:
    r: Scalar
    phi: Scalar

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        del flavor
        return jnp.asarray(self.r) * _phase(self.phi)


@dataclass(frozen=True)
class RealImag:
    """CP-conserving coefficient ``x + i y``.

    ``x`` and ``y`` may be plain numerical values or fit ``Parameter`` objects.
    When fit parameters are supplied, ``value(..., values=mapping)`` resolves
    their current values from the minimizer mapping.
    """

    x: object
    y: object

    @property
    def parameters(self) -> tuple[object, ...]:
        return tuple(value for value in (self.x, self.y) if hasattr(value, "resolve"))

    def value(
        self,
        flavor: Flavor = Flavor.PARTICLE,
        values: Mapping[str, object] | None = None,
    ) -> Array:
        del flavor
        x = _resolve(self.x, values)
        y = _resolve(self.y, values)
        return jnp.asarray(x) + 1j * jnp.asarray(y)


@dataclass(frozen=True)
class BelleCP:
    a: Scalar
    b: Scalar
    delta: Scalar
    phi: Scalar

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        return jnp.asarray(self.a) * _phase(self.delta) * (
            1.0 + flavor.sign * jnp.asarray(self.b) * _phase(self.phi)
        )


@dataclass(frozen=True)
class CartesianCP:
    x: Scalar
    y: Scalar
    dx: Scalar
    dy: Scalar

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        sign = flavor.sign
        return jnp.asarray(self.x) + sign * jnp.asarray(self.dx) + 1j * (
            jnp.asarray(self.y) + sign * jnp.asarray(self.dy)
        )


@dataclass(frozen=True)
class CartesianGammaCP:
    x: Scalar
    y: Scalar
    x_cp: Scalar
    y_cp: Scalar
    delta_x_cp: Scalar
    delta_y_cp: Scalar

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        sign = flavor.sign
        base = jnp.asarray(self.x) + 1j * jnp.asarray(self.y)
        correction = (
            1.0
            + jnp.asarray(self.x_cp)
            + sign * jnp.asarray(self.delta_x_cp)
            + 1j
            * (jnp.asarray(self.y_cp) + sign * jnp.asarray(self.delta_y_cp))
        )
        return base * correction


@dataclass(frozen=True)
class CleoCP:
    a: Scalar
    b: Scalar
    delta: Scalar
    phi: Scalar

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        sign = flavor.sign
        magnitude = jnp.asarray(self.a) + sign * jnp.asarray(self.b)
        angle = jnp.asarray(self.delta) + sign * jnp.asarray(self.phi)
        return magnitude * _phase(angle)


@dataclass(frozen=True)
class MagPhaseCP:
    r: Scalar
    phi: Scalar
    r_bar: Scalar
    phi_bar: Scalar

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        if flavor is Flavor.PARTICLE:
            return jnp.asarray(self.r) * _phase(self.phi)
        return jnp.asarray(self.r_bar) * _phase(self.phi_bar)


@dataclass(frozen=True)
class PolarGammaCP:
    x: Scalar
    y: Scalar
    r: Scalar
    delta: Scalar
    gamma: Scalar

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        sign = flavor.sign
        base = jnp.asarray(self.x) + 1j * jnp.asarray(self.y)
        phase = jnp.asarray(self.delta) + sign * jnp.asarray(self.gamma)
        return base * (1.0 + jnp.asarray(self.r) * _phase(phase))


@dataclass(frozen=True)
class RealImagCP:
    x: Scalar
    y: Scalar
    x_bar: Scalar
    y_bar: Scalar

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        if flavor is Flavor.PARTICLE:
            return jnp.asarray(self.x) + 1j * jnp.asarray(self.y)
        return jnp.asarray(self.x_bar) + 1j * jnp.asarray(self.y_bar)


@dataclass(frozen=True)
class RealImagGammaCP:
    x: Scalar
    y: Scalar
    x_cp: Scalar
    y_cp: Scalar
    x_cp_bar: Scalar
    y_cp_bar: Scalar

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        base = jnp.asarray(self.x) + 1j * jnp.asarray(self.y)
        if flavor is Flavor.PARTICLE:
            correction = 1.0 + jnp.asarray(self.x_cp) + 1j * jnp.asarray(self.y_cp)
        else:
            correction = (
                1.0
                + jnp.asarray(self.x_cp_bar)
                + 1j * jnp.asarray(self.y_cp_bar)
            )
        return base * correction
