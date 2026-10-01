"""Charge-dependent complex coefficients for direct CP-violation fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

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
        """Embedded fit ``Parameter`` objects among ``x, y, dx, dy``."""

        return tuple(
            value
            for value in (self.x, self.y, self.dx, self.dy)
            if hasattr(value, "resolve")
        )

    def value(self, values: Mapping[str, object] | None = None):
        """Resolve the complex coefficient for this instance's `charge`."""

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

    @property
    def real_part(self) -> "CPRealImagPart":
        """Real part of `value`, resolvable like a scalar fit parameter.

        Intended for per-knot CP violation in ``QMI(real_parts=...)``: build one
        ``CPRealImag`` per knot, call ``for_charge(q)`` and pass
        ``coefficient.real_part`` / ``coefficient.imag_part`` for that charge.
        """

        return CPRealImagPart(self, "real")

    @property
    def imag_part(self) -> "CPRealImagPart":
        """Imaginary part of `value`; see `real_part`."""

        return CPRealImagPart(self, "imag")


@dataclass(frozen=True)
class CPRealImagPart:
    """Real or imaginary part of a `CPRealImag` for its own `charge`.

    Exposes ``parameters`` and ``resolve(values)`` so it can stand wherever a
    plugin accepts a scalar fit parameter (e.g. ``QMI`` Cartesian knots).
    """

    coefficient: CPRealImag
    part: Literal["real", "imag"]

    def __post_init__(self) -> None:
        if self.part not in ("real", "imag"):
            raise ValueError("part must be 'real' or 'imag'")

    @property
    def parameters(self) -> tuple[object, ...]:
        return self.coefficient.parameters

    def resolve(self, values: Mapping[str, object] | None = None):
        value = self.coefficient.value(values)
        return jnp.real(value) if self.part == "real" else jnp.imag(value)


__all__ = ["CPRealImag"]
