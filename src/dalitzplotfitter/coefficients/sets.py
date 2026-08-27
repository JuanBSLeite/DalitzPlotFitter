"""Complex amplitude coefficient parameterization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp


def _resolve(value: object, values: Mapping[str, object] | None = None):
    resolver = getattr(value, "resolve", None)
    return resolver(values) if resolver is not None else value


@dataclass(frozen=True)
class RealImag:
    """Complex coefficient ``c = x + i y``.

    ``x`` and ``y`` may be numerical constants or fit ``Parameter`` objects.
    """

    x: object
    y: object

    @property
    def parameters(self) -> tuple[object, ...]:
        return tuple(value for value in (self.x, self.y) if hasattr(value, "resolve"))

    def value(self, values: Mapping[str, object] | None = None):
        return jnp.asarray(_resolve(self.x, values)) + 1j * jnp.asarray(
            _resolve(self.y, values)
        )
