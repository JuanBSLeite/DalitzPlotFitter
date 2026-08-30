"""Quasi-model-independent S-wave parameterisation.

The default convention follows the LHCb D+ -> pi- pi+ pi+ QMIPWA:
individual S-wave magnitudes and phases are specified at mass knots and
interpolated linearly as functions of the two-body invariant mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .context import ResonanceContext


@dataclass(frozen=True)
class QMI:
    """Quasi-model-independent scalar amplitude defined at mass knots.

    Parameters
    ----------
    knots:
        Strictly increasing two-body invariant masses in GeV.
    magnitudes:
        S-wave magnitudes at the knots. Entries may be numerical constants or
        fit ``Parameter`` objects.
    phases:
        S-wave phases at the knots, in radians. Entries may be numerical
        constants or fit ``Parameter`` objects.

    Notes
    -----
    The amplitude is

    ``A(m) = a(m) exp(i delta(m))``

    where ``a`` and ``delta`` are interpolated linearly between neighbouring
    knots. Outside the knot range the nearest endpoint value is used. In a
    physical Dalitz model the first and last knots should normally cover the
    complete kinematic mass range of the selected pair.
    """

    knots: tuple[float, ...]
    magnitudes: tuple[object, ...]
    phases: tuple[object, ...]

    def __post_init__(self) -> None:
        if len(self.knots) < 2:
            raise ValueError("QMI requires at least two knots")
        if len(self.magnitudes) != len(self.knots):
            raise ValueError("QMI magnitudes must have the same length as knots")
        if len(self.phases) != len(self.knots):
            raise ValueError("QMI phases must have the same length as knots")
        knots = tuple(float(value) for value in self.knots)
        if any(right <= left for left, right in zip(knots[:-1], knots[1:])):
            raise ValueError("QMI knots must be strictly increasing")
        if knots[0] <= 0.0:
            raise ValueError("QMI knot masses must be positive")

    @property
    def size(self) -> int:
        return len(self.knots)

    def interpolated_magnitude_phase(self, mass):
        mass = jnp.asarray(mass)
        knots = jnp.asarray(self.knots, dtype=mass.dtype)
        magnitudes = jnp.asarray(self.magnitudes, dtype=mass.dtype)
        phases = jnp.asarray(self.phases, dtype=mass.dtype)
        magnitude = jnp.interp(mass, knots, magnitudes)
        phase = jnp.interp(mass, knots, phases)
        return magnitude, phase

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("QMI is defined for a scalar S-wave")
        magnitude, phase = self.interpolated_magnitude_phase(mass)
        return magnitude * jnp.exp(1j * phase)


__all__ = ["QMI"]
