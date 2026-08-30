"""Quasi-model-independent S-wave parameterisation.

The default convention follows the LHCb QMIPWA construction used for
D_s+ -> pi- pi+ pi+: individual S-wave magnitudes and phases are specified at
fixed pi-pi mass points, while the linear splines are evaluated as functions of
s = m(pi pi)^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .context import ResonanceContext


@dataclass(frozen=True)
class QMI:
    """Quasi-model-independent scalar amplitude defined at mass knots.

    ``knots`` are supplied as two-body invariant masses in GeV for convenience,
    matching the published LHCb tables. Internally, magnitude and phase are
    linearly interpolated in ``s = m**2``:

    ``A(s) = a(s) exp(i delta(s))``.

    Entries of ``magnitudes`` and ``phases`` may be numerical constants or fit
    ``Parameter`` objects. Phases are in radians and should be supplied as a
    continuous/unwrapped sequence.
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
        knot_masses = jnp.asarray(self.knots, dtype=mass.dtype)
        knot_s = knot_masses**2
        s = mass**2
        magnitudes = jnp.asarray(self.magnitudes, dtype=mass.dtype)
        phases = jnp.asarray(self.phases, dtype=mass.dtype)
        magnitude = jnp.interp(s, knot_s, magnitudes)
        phase = jnp.interp(s, knot_s, phases)
        return magnitude, phase

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("QMI is defined for a scalar S-wave")
        magnitude, phase = self.interpolated_magnitude_phase(mass)
        return magnitude * jnp.exp(1j * phase)


__all__ = ["QMI"]
