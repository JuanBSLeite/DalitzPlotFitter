"""Charge-dependent complex coefficients for direct CP-violation fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp


def _resolve(value: object, values: Mapping[str, object] | None = None):
    resolver = getattr(value, "resolve", None)
    return resolver(values) if resolver is not None else value


@dataclass(frozen=True)
class CPRealImag:
    """Cartesian CP coefficient shared between charge-conjugate samples.

    The two charge states are

    ``c_q = (x + q*dx) + i (y + q*dy)``, with ``q = +1`` or ``-1``.

    ``x`` and ``y`` are the CP-averaged Cartesian coefficient components,
    while ``dx`` and ``dy`` parameterize the direct-CP difference. All four
    entries may be numerical constants or fit ``Parameter`` objects.
    """

    x: object
    y: object
    dx: object = 0.0
    dy: object = 0.0
    charge: int = +1

    def __post_init__(self) -> None:
        if self.charge not in (-1, +1):
            raise ValueError("CPRealImag charge must be +1 or -1")

    @property
    def parameters(self) -> tuple[object, ...]:
        return tuple(
            value
            for value in (self.x, self.y, self.dx, self.dy)
            if hasattr(value, "resolve")
        )

    def value(self, values: Mapping[str, object] | None = None):
        q = float(self.charge)
        real = jnp.asarray(_resolve(self.x, values)) + q * jnp.asarray(
            _resolve(self.dx, values)
        )
        imag = jnp.asarray(_resolve(self.y, values)) + q * jnp.asarray(
            _resolve(self.dy, values)
        )
        return real + 1j * imag

    def for_charge(self, charge: int) -> "CPRealImag":
        """Return the same shared parameterization for the requested charge."""

        return CPRealImag(self.x, self.y, self.dx, self.dy, charge=charge)


__all__ = ["CPRealImag"]
