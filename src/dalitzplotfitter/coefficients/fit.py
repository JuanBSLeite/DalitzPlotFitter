"""Fit-aware coefficient parameterizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp

from dalitzplotfitter.fit import Parameter

from .base import Flavor


@dataclass(frozen=True)
class FitMagPhase:
    """Magnitude/phase coefficient controlled by fit parameters."""

    r: Parameter
    phi: Parameter

    @property
    def parameters(self) -> tuple[Parameter, Parameter]:
        return (self.r, self.phi)

    def value(
        self,
        flavor: Flavor = Flavor.PARTICLE,
        values: Mapping[str, object] | None = None,
    ):
        del flavor
        r = self.r.resolve(values)
        phi = self.phi.resolve(values)
        return jnp.asarray(r) * jnp.exp(1j * jnp.asarray(phi))


@dataclass(frozen=True)
class FitCartesianCP:
    """Cartesian CP coefficient controlled by fit parameters."""

    x: Parameter
    y: Parameter
    dx: Parameter
    dy: Parameter

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return (self.x, self.y, self.dx, self.dy)

    def value(
        self,
        flavor: Flavor = Flavor.PARTICLE,
        values: Mapping[str, object] | None = None,
    ):
        sign = flavor.sign
        x = self.x.resolve(values)
        y = self.y.resolve(values)
        dx = self.dx.resolve(values)
        dy = self.dy.resolve(values)
        return jnp.asarray(x) + sign * jnp.asarray(dx) + 1j * (
            jnp.asarray(y) + sign * jnp.asarray(dy)
        )
