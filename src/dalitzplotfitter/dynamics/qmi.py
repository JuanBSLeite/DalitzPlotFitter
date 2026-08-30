"""Quasi-model-independent S-wave parameterisation.

The default convention follows the LHCb QMIPWA construction used for
D_s+ -> pi- pi+ pi+: individual S-wave magnitudes and phases are specified at
fixed pi-pi mass points, and the interpolation coordinate is s = m(pi pi)^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .context import ResonanceContext


def _natural_cubic_spline(x, xp, fp):
    """Evaluate a differentiable natural cubic spline with endpoint clamping."""

    x = jnp.asarray(x)
    xp = jnp.asarray(xp, dtype=x.dtype)
    fp = jnp.asarray(fp, dtype=x.dtype)
    n = xp.shape[0]

    h = xp[1:] - xp[:-1]
    interior = n - 2

    # Natural boundary conditions: second derivative vanishes at endpoints.
    if interior > 0:
        diag = 2.0 * (h[:-1] + h[1:])
        rhs = 6.0 * ((fp[2:] - fp[1:-1]) / h[1:] - (fp[1:-1] - fp[:-2]) / h[:-1])
        matrix = jnp.diag(diag)
        if interior > 1:
            off = h[1:-1]
            matrix = matrix + jnp.diag(off, 1) + jnp.diag(off, -1)
        second_inner = jnp.linalg.solve(matrix, rhs)
        second = jnp.concatenate((jnp.zeros(1, dtype=x.dtype), second_inner, jnp.zeros(1, dtype=x.dtype)))
    else:
        second = jnp.zeros(n, dtype=x.dtype)

    x_clamped = jnp.clip(x, xp[0], xp[-1])
    index = jnp.searchsorted(xp, x_clamped, side="right") - 1
    index = jnp.clip(index, 0, n - 2)

    x0 = xp[index]
    x1 = xp[index + 1]
    y0 = fp[index]
    y1 = fp[index + 1]
    m0 = second[index]
    m1 = second[index + 1]
    width = x1 - x0
    a = (x1 - x_clamped) / width
    b = (x_clamped - x0) / width

    return (
        a * y0
        + b * y1
        + ((a**3 - a) * m0 + (b**3 - b) * m1) * width**2 / 6.0
    )


@dataclass(frozen=True)
class QMI:
    """Quasi-model-independent scalar amplitude defined at mass knots.

    ``knots`` are supplied as two-body invariant masses in GeV. Internally,
    magnitude and phase are interpolated in ``s = m**2``:

    ``A(s) = a(s) exp(i delta(s))``.

    ``interpolation`` may be ``"linear"`` (the default, matching the published
    LHCb QMIPWA convention) or ``"cubic"`` (a natural cubic spline). Magnitude
    and phase are always interpolated separately. Entries of ``magnitudes`` and
    ``phases`` may be numerical constants or fit ``Parameter`` objects. Phases
    are in radians and should be supplied as a continuous/unwrapped sequence.
    """

    knots: tuple[float, ...]
    magnitudes: tuple[object, ...]
    phases: tuple[object, ...]
    interpolation: str = "linear"

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
        if self.interpolation not in {"linear", "cubic"}:
            raise ValueError("QMI interpolation must be 'linear' or 'cubic'")
        if self.interpolation == "cubic" and len(self.knots) < 3:
            raise ValueError("cubic QMI interpolation requires at least three knots")

    @property
    def size(self) -> int:
        return len(self.knots)

    def _interpolate(self, s, knot_s, values):
        if self.interpolation == "linear":
            return jnp.interp(s, knot_s, values)
        return _natural_cubic_spline(s, knot_s, values)

    def interpolated_magnitude_phase(self, mass):
        mass = jnp.asarray(mass)
        knot_masses = jnp.asarray(self.knots, dtype=mass.dtype)
        knot_s = knot_masses**2
        s = mass**2
        magnitudes = jnp.asarray(self.magnitudes, dtype=mass.dtype)
        phases = jnp.asarray(self.phases, dtype=mass.dtype)
        magnitude = self._interpolate(s, knot_s, magnitudes)
        phase = self._interpolate(s, knot_s, phases)
        return magnitude, phase

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("QMI is defined for a scalar S-wave")
        magnitude, phase = self.interpolated_magnitude_phase(mass)
        return magnitude * jnp.exp(1j * phase)


__all__ = ["QMI"]
