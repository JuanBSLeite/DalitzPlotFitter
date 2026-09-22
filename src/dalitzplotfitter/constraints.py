"""External likelihood constraints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from dalitzplotfitter.dynamics.lineshape.qmi import QMI

Parameters = Mapping[str, object]


def _resolve(value: object, parameters: Parameters):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


@dataclass(frozen=True)
class GaussianConstraint:
    """Gaussian penalty ``0.5*((x-mu)/sigma)^2`` up to an additive constant."""

    parameter: object
    mean: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("GaussianConstraint sigma must be positive")

    def __call__(self, parameters: Parameters) -> Array:
        value = jnp.asarray(_resolve(self.parameter, parameters))
        return 0.5 * ((value - self.mean) / self.sigma) ** 2


@dataclass(frozen=True)
class QMISmoothnessConstraint:
    """Optional discrete curvature penalty on complex 1D QMI knot values.

    For ``s_i = knots[i]**2``, ``h_i = s[i+1] - s[i]`` and complex
    slopes ``d_i = (q[i+1] - q[i]) / h_i``, add to the *total* NLL::

        strength * sum(weights[i] * 2 * abs(d[i+1] - d[i])**2
                       / (h[i+1] + h[i]))

    ``weights`` has one nonnegative entry per interior knot, defaulting to
    one. A zero disables that entire three-knot stencil. Two knots have no
    interior curvature and give zero. This is a discrete approximation to
    integrated squared curvature in s, NOT the integral of the interpolant's
    squared second derivative (which would be unsuitable for linear QMI).
    It is identical for all interpolation modes and evaluates polar nodes
    as ``magnitude * exp(1j * phase)`` before differencing.

    Strength is a fixed hyperparameter, never a fit Parameter. No factor of
    1/2, division by event count, or amplitude normalization is implicit.
    The raw penalty has units of amplitude squared / GeV**6 for GeV knots.
    Choose a fixed QMI scale convention before using it: a freely rescalable
    or dynamically unit-normalized QMI could otherwise evade the penalty by
    shrinking its raw nodes. It does not modify the amplitude or PDF itself.
    """

    qmi: QMI
    strength: float = 0.0
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        from dalitzplotfitter.dynamics.lineshape.qmi import QMI

        if not isinstance(self.qmi, QMI):
            raise TypeError("QMISmoothnessConstraint requires a 1D QMI")
        try:
            strength = float(self.strength)
        except (TypeError, ValueError) as exc:
            raise ValueError("QMI smoothness strength must be a fixed number") from exc
        if not isfinite(strength) or strength < 0:
            raise ValueError("QMI smoothness strength must be finite and nonnegative")
        if self.qmi.interpolation == "none" and strength != 0:
            raise ValueError(
                "QMI smoothness requires interpolated knots; "
                "use strength=0 for constant bins"
            )
        object.__setattr__(self, "strength", strength)
        if self.weights is not None:
            try:
                weights = tuple(float(weight) for weight in self.weights)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "QMI smoothness weights must be fixed numbers"
                ) from exc
            if len(weights) != self.qmi.size - 2:
                raise ValueError("QMI smoothness requires one weight per interior knot")
            if any(not isfinite(weight) or weight < 0 for weight in weights):
                raise ValueError(
                    "QMI smoothness weights must be finite and nonnegative"
                )
            object.__setattr__(self, "weights", weights)

    def __call__(self, parameters: Parameters | None = None) -> Array:
        if self.strength == 0 or self.qmi.size == 2:
            return jnp.asarray(0.0)
        if self.weights is not None and not any(self.weights):
            return jnp.asarray(0.0)

        def resolve(group):
            return jnp.asarray([_resolve(value, parameters) for value in group])

        if self.qmi.parameterization == "cartesian":
            real = resolve(self.qmi.real_parts)
            imaginary = resolve(self.qmi.imaginary_parts)
        else:
            magnitude = resolve(self.qmi.magnitudes)
            phase = resolve(self.qmi.phases)
            real = magnitude * jnp.cos(phase)
            imaginary = magnitude * jnp.sin(phase)
        widths = jnp.diff(jnp.asarray(self.qmi.knots) ** 2)
        real_change = jnp.diff(jnp.diff(real) / widths)
        imaginary_change = jnp.diff(jnp.diff(imaginary) / widths)
        curvature = 2 * (real_change**2 + imaginary_change**2)
        curvature = curvature / (widths[:-1] + widths[1:])
        if self.weights is not None:
            curvature = curvature * jnp.asarray(self.weights)
        return self.strength * jnp.sum(curvature)


@dataclass(frozen=True)
class ConstrainedNLL:
    """Add one or more external constraints to an existing NLL."""

    nll: object
    constraints: tuple[object, ...]

    def __init__(self, nll: object, *constraints: object):
        object.__setattr__(self, "nll", nll)
        object.__setattr__(self, "constraints", tuple(constraints))

    def __call__(self, parameters: Parameters) -> Array:
        total = jnp.asarray(self.nll(parameters))
        for constraint in self.constraints:
            total = total + jnp.asarray(constraint(parameters))
        return total


__all__ = ["ConstrainedNLL", "GaussianConstraint", "QMISmoothnessConstraint"]
