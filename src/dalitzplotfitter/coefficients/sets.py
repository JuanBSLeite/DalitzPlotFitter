"""Complex coefficient parameterisations from Laura++ Table 1."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from .base import Flavor

Scalar = float | Array


def _phase(angle: Scalar) -> Array:
    return jnp.exp(1j * jnp.asarray(angle))


@dataclass(frozen=True)
class MagPhase:
    r: Scalar
    phi: Scalar

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        del flavor
        return jnp.asarray(self.r) * _phase(self.phi)


@dataclass(frozen=True)
class RealImag:
    x: Scalar
    y: Scalar

    def value(self, flavor: Flavor = Flavor.PARTICLE) -> Array:
        del flavor
        return jnp.asarray(self.x) + 1j * jnp.asarray(self.y)


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
