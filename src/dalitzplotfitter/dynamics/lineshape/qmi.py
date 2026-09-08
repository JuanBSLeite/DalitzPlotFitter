"""Quasi-model-independent S-wave parameterisation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

from ..context import ResonanceContext


@jax.custom_vjp
def _linear_qmi_prepared(
    magnitudes,
    phases,
    index,
    fraction,
    order,
    starts,
    ends,
):
    """Evaluate prepared linear QMI interpolation with a reduction-based VJP.

    The ordinary reverse-mode derivative of indexed interpolation is expressed
    by XLA as large scatter-add operations.  On consumer GPUs, FP64 atomics can
    dominate the fit time by orders of magnitude.  Because linear QMI support
    is local, each event contributes only to the two knots bordering its fixed
    interval.  The custom VJP accumulates those contributions after grouping
    events by interval, avoiding the large reverse scatter while preserving the
    exact forward model.
    """

    index32 = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction, dtype=magnitudes.dtype)
    magnitude = magnitudes[index32] + fraction * (
        magnitudes[index32 + 1] - magnitudes[index32]
    )
    phase = phases[index32] + fraction * (phases[index32 + 1] - phases[index32])
    return magnitude * jnp.exp(1j * phase)


def _linear_qmi_prepared_fwd(
    magnitudes,
    phases,
    index,
    fraction,
    order,
    starts,
    ends,
):
    index32 = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction, dtype=magnitudes.dtype)
    magnitude = magnitudes[index32] + fraction * (
        magnitudes[index32 + 1] - magnitudes[index32]
    )
    phase = phases[index32] + fraction * (phases[index32 + 1] - phases[index32])
    exp_phase = jnp.exp(1j * phase)
    value = magnitude * exp_phase
    residual = (value, exp_phase, fraction, order, starts, ends)
    return value, residual


def _grouped_interval_sums(values, starts, ends):
    """Sum a sorted value vector over the fixed QMI interpolation intervals."""

    prefix = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=values.dtype),
            jnp.cumsum(values),
        )
    )
    starts = jnp.asarray(starts, dtype=jnp.int32)
    ends = jnp.asarray(ends, dtype=jnp.int32)
    return prefix[ends] - prefix[starts]


def _linear_qmi_prepared_bwd(residual, cotangent):
    value, exp_phase, fraction, order, starts, ends = residual

    # JAX's real-parameter/complex-output VJP convention is
    # dL/dx = Re(g * dy/dx), where g is the incoming complex cotangent.
    d_magnitude = jnp.real(cotangent * exp_phase)
    d_phase = jnp.real(cotangent * (1j * value))

    order = jnp.asarray(order, dtype=jnp.int32)
    sorted_fraction = fraction[order]
    sorted_d_magnitude = d_magnitude[order]
    sorted_d_phase = d_phase[order]

    left_magnitude = _grouped_interval_sums(
        (1.0 - sorted_fraction) * sorted_d_magnitude,
        starts,
        ends,
    )
    right_magnitude = _grouped_interval_sums(
        sorted_fraction * sorted_d_magnitude,
        starts,
        ends,
    )
    left_phase = _grouped_interval_sums(
        (1.0 - sorted_fraction) * sorted_d_phase,
        starts,
        ends,
    )
    right_phase = _grouped_interval_sums(
        sorted_fraction * sorted_d_phase,
        starts,
        ends,
    )

    magnitude_gradient = jnp.concatenate(
        (
            left_magnitude[:1],
            left_magnitude[1:] + right_magnitude[:-1],
            right_magnitude[-1:],
        )
    )
    phase_gradient = jnp.concatenate(
        (
            left_phase[:1],
            left_phase[1:] + right_phase[:-1],
            right_phase[-1:],
        )
    )

    return (
        magnitude_gradient,
        phase_gradient,
        None,
        None,
        None,
        None,
        None,
    )


_linear_qmi_prepared.defvjp(
    _linear_qmi_prepared_fwd,
    _linear_qmi_prepared_bwd,
)


@jax.custom_vjp
def _linear_cartesian_qmi_prepared(
    real_parts,
    imaginary_parts,
    index,
    fraction,
    order,
    starts,
    ends,
):
    """Evaluate a prepared Cartesian QMI with a reduction-based VJP."""

    index32 = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction, dtype=real_parts.dtype)
    real = real_parts[index32] + fraction * (
        real_parts[index32 + 1] - real_parts[index32]
    )
    imaginary = imaginary_parts[index32] + fraction * (
        imaginary_parts[index32 + 1] - imaginary_parts[index32]
    )
    return real + 1j * imaginary


def _linear_cartesian_qmi_prepared_fwd(
    real_parts,
    imaginary_parts,
    index,
    fraction,
    order,
    starts,
    ends,
):
    index32 = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction, dtype=real_parts.dtype)
    real = real_parts[index32] + fraction * (
        real_parts[index32 + 1] - real_parts[index32]
    )
    imaginary = imaginary_parts[index32] + fraction * (
        imaginary_parts[index32 + 1] - imaginary_parts[index32]
    )
    value = real + 1j * imaginary
    residual = (fraction, order, starts, ends)
    return value, residual


def _linear_cartesian_qmi_prepared_bwd(residual, cotangent):
    fraction, order, starts, ends = residual

    # JAX's real-parameter/complex-output VJP convention is
    # dL/dx = Re(g * dy/dx).
    d_real = jnp.real(cotangent)
    d_imaginary = jnp.real(1j * cotangent)

    order = jnp.asarray(order, dtype=jnp.int32)
    sorted_fraction = fraction[order]
    sorted_d_real = d_real[order]
    sorted_d_imaginary = d_imaginary[order]

    left_real = _grouped_interval_sums(
        (1.0 - sorted_fraction) * sorted_d_real,
        starts,
        ends,
    )
    right_real = _grouped_interval_sums(
        sorted_fraction * sorted_d_real,
        starts,
        ends,
    )
    left_imaginary = _grouped_interval_sums(
        (1.0 - sorted_fraction) * sorted_d_imaginary,
        starts,
        ends,
    )
    right_imaginary = _grouped_interval_sums(
        sorted_fraction * sorted_d_imaginary,
        starts,
        ends,
    )

    real_gradient = jnp.concatenate(
        (
            left_real[:1],
            left_real[1:] + right_real[:-1],
            right_real[-1:],
        )
    )
    imaginary_gradient = jnp.concatenate(
        (
            left_imaginary[:1],
            left_imaginary[1:] + right_imaginary[:-1],
            right_imaginary[-1:],
        )
    )

    return (
        real_gradient,
        imaginary_gradient,
        None,
        None,
        None,
        None,
        None,
    )


_linear_cartesian_qmi_prepared.defvjp(
    _linear_cartesian_qmi_prepared_fwd,
    _linear_cartesian_qmi_prepared_bwd,
)


def _assemble_interval_endpoint_sums(left, right):
    """Assemble per-interval left/right sums into per-knot gradients."""

    return jnp.concatenate(
        (
            left[:1],
            left[1:] + right[:-1],
            right[-1:],
        )
    )


def _cubic_qmi_prepared_impl(
    values,
    index,
    fraction,
    knot_s,
    inverse,
):
    """Evaluate a prepared natural cubic spline with fixed knot geometry."""

    values = jnp.asarray(values)
    index32 = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction, dtype=values.dtype)
    knot_s = jnp.asarray(knot_s, dtype=values.dtype)
    inverse = jnp.asarray(inverse, dtype=values.dtype)

    h = knot_s[1:] - knot_s[:-1]
    rhs = 6.0 * (
        (values[2:] - values[1:-1]) / h[1:]
        - (values[1:-1] - values[:-2]) / h[:-1]
    )
    second_inner = inverse @ rhs
    second = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=values.dtype),
            second_inner,
            jnp.zeros((1,), dtype=values.dtype),
        )
    )

    width = knot_s[index32 + 1] - knot_s[index32]
    a = 1.0 - fraction
    b = fraction
    left_second_weight = (a**3 - a) * width**2 / 6.0
    right_second_weight = (b**3 - b) * width**2 / 6.0
    return (
        a * values[index32]
        + b * values[index32 + 1]
        + left_second_weight * second[index32]
        + right_second_weight * second[index32 + 1]
    )


@jax.custom_vjp
def _cubic_qmi_prepared(
    values,
    index,
    fraction,
    order,
    starts,
    ends,
    knot_s,
    inverse,
):
    """Prepared natural cubic spline with a reduction-based reverse pass.

    The forward expression is exactly the ordinary natural cubic spline.  In
    reverse mode, event contributions are first reduced by their fixed
    interpolation interval, just like the optimized linear QMI path.  The
    global spline coupling is then propagated only in knot space through the
    small natural-spline system, avoiding event-sized scatter-add operations.
    """

    return _cubic_qmi_prepared_impl(
        values,
        index,
        fraction,
        knot_s,
        inverse,
    )


def _cubic_qmi_prepared_fwd(
    values,
    index,
    fraction,
    order,
    starts,
    ends,
    knot_s,
    inverse,
):
    value = _cubic_qmi_prepared_impl(
        values,
        index,
        fraction,
        knot_s,
        inverse,
    )
    residual = (
        index,
        fraction,
        order,
        starts,
        ends,
        knot_s,
        inverse,
    )
    return value, residual


def _cubic_qmi_prepared_bwd(residual, cotangent):
    (
        index,
        fraction,
        order,
        starts,
        ends,
        knot_s,
        inverse,
    ) = residual

    index = jnp.asarray(index, dtype=jnp.int32)
    fraction = jnp.asarray(fraction)
    order = jnp.asarray(order, dtype=jnp.int32)
    knot_s = jnp.asarray(knot_s, dtype=fraction.dtype)
    inverse = jnp.asarray(inverse, dtype=fraction.dtype)
    cotangent = jnp.asarray(cotangent, dtype=fraction.dtype)

    width = knot_s[index + 1] - knot_s[index]
    a = 1.0 - fraction
    b = fraction
    left_second_weight = (a**3 - a) * width**2 / 6.0
    right_second_weight = (b**3 - b) * width**2 / 6.0

    sorted_cotangent = cotangent[order]
    sorted_a = a[order]
    sorted_b = b[order]
    sorted_left_second_weight = left_second_weight[order]
    sorted_right_second_weight = right_second_weight[order]

    direct_left = _grouped_interval_sums(
        sorted_a * sorted_cotangent,
        starts,
        ends,
    )
    direct_right = _grouped_interval_sums(
        sorted_b * sorted_cotangent,
        starts,
        ends,
    )
    direct_gradient = _assemble_interval_endpoint_sums(
        direct_left,
        direct_right,
    )

    second_left = _grouped_interval_sums(
        sorted_left_second_weight * sorted_cotangent,
        starts,
        ends,
    )
    second_right = _grouped_interval_sums(
        sorted_right_second_weight * sorted_cotangent,
        starts,
        ends,
    )
    second_gradient = _assemble_interval_endpoint_sums(
        second_left,
        second_right,
    )

    # Natural boundary conditions fix the endpoint second derivatives to zero,
    # so only the interior second-derivative cotangents propagate through the
    # fixed spline system: M_inner = inverse @ rhs(values).
    rhs_gradient = inverse.T @ second_gradient[1:-1]

    h = knot_s[1:] - knot_s[:-1]
    rhs_left = 6.0 * rhs_gradient / h[:-1]
    rhs_center = -6.0 * rhs_gradient * (1.0 / h[:-1] + 1.0 / h[1:])
    rhs_right = 6.0 * rhs_gradient / h[1:]
    spline_gradient = (
        jnp.pad(rhs_left, (0, 2))
        + jnp.pad(rhs_center, (1, 1))
        + jnp.pad(rhs_right, (2, 0))
    )

    values_gradient = direct_gradient + spline_gradient
    return (
        values_gradient,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_cubic_qmi_prepared.defvjp(
    _cubic_qmi_prepared_fwd,
    _cubic_qmi_prepared_bwd,
)


def _interval_index_and_fraction(x, xp, prepared_index=None):
    """Return the interpolation interval and local fraction.

    A prepared index lets large repeated QMI evaluations reuse the fixed knot
    lookup while recomputing only the inexpensive local fraction.
    """

    x = jnp.asarray(x)
    xp = jnp.asarray(xp, dtype=x.dtype)
    x_clamped = jnp.clip(x, xp[0], xp[-1])
    if prepared_index is None:
        index = jnp.clip(
            jnp.searchsorted(xp, x_clamped, side="right") - 1,
            0,
            xp.shape[0] - 2,
        )
    else:
        index = jnp.asarray(prepared_index, dtype=jnp.int32)
        index = jnp.clip(index, 0, xp.shape[0] - 2)
    x0 = xp[index]
    x1 = xp[index + 1]
    fraction = (x_clamped - x0) / (x1 - x0)
    return index, fraction


def _linear_spline(x, xp, fp):
    """Piecewise-linear interpolation with endpoint clamping.

    This is implemented directly rather than through jax.numpy.interp.
    QMI evaluations happen on very large normalization grids and the extra
    interpolation primitive can make JAX tracing/compilation disproportionately
    slow. The explicit form uses the same search/index pattern as the cubic spline.
    """

    x = jnp.asarray(x)
    xp = jnp.asarray(xp, dtype=x.dtype)
    fp = jnp.asarray(fp, dtype=x.dtype)

    index, fraction = _interval_index_and_fraction(x, xp)
    y0, y1 = fp[index], fp[index + 1]
    return y0 + fraction * (y1 - y0)


def _natural_cubic_spline(x, xp, fp, inverse=None):
    x = jnp.asarray(x)
    xp = jnp.asarray(xp, dtype=x.dtype)
    fp = jnp.asarray(fp, dtype=x.dtype)
    n = xp.shape[0]
    h = xp[1:] - xp[:-1]
    interior = n - 2
    if interior > 0:
        rhs = 6.0 * ((fp[2:] - fp[1:-1]) / h[1:] - (fp[1:-1] - fp[:-2]) / h[:-1])
        if inverse is None:
            diag = 2.0 * (h[:-1] + h[1:])
            matrix = jnp.diag(diag)
            if interior > 1:
                off = h[1:-1]
                matrix = matrix + jnp.diag(off, 1) + jnp.diag(off, -1)
            second_inner = jnp.linalg.solve(matrix, rhs)
        else:
            second_inner = jnp.asarray(inverse, dtype=x.dtype) @ rhs
        second = jnp.concatenate(
            (
                jnp.zeros(1, dtype=x.dtype),
                second_inner,
                jnp.zeros(1, dtype=x.dtype),
            )
        )
    else:
        second = jnp.zeros(n, dtype=x.dtype)
    x_clamped = jnp.clip(x, xp[0], xp[-1])
    index = jnp.clip(
        jnp.searchsorted(xp, x_clamped, side="right") - 1,
        0,
        n - 2,
    )
    x0, x1 = xp[index], xp[index + 1]
    y0, y1 = fp[index], fp[index + 1]
    m0, m1 = second[index], second[index + 1]
    width = x1 - x0
    a = (x1 - x_clamped) / width
    b = (x_clamped - x0) / width
    return a * y0 + b * y1 + ((a**3 - a) * m0 + (b**3 - b) * m1) * width**2 / 6.0


@dataclass(frozen=True)
class QMI:
    knots: tuple[float, ...]
    magnitudes: tuple[object, ...] | None = None
    phases: tuple[object, ...] | None = None
    interpolation: str = "linear"
    real_parts: tuple[object, ...] | None = None
    imaginary_parts: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.knots) < 2:
            raise ValueError("QMI requires at least two knots")
        polar = self.magnitudes is not None or self.phases is not None
        cartesian = self.real_parts is not None or self.imaginary_parts is not None
        if polar == cartesian:
            raise ValueError(
                "QMI requires exactly one complete polar or Cartesian parameter set"
            )
        if polar:
            if self.magnitudes is None or len(self.magnitudes) != len(self.knots):
                raise ValueError("QMI magnitudes must have the same length as knots")
            if self.phases is None or len(self.phases) != len(self.knots):
                raise ValueError("QMI phases must have the same length as knots")
        else:
            if self.real_parts is None or len(self.real_parts) != len(self.knots):
                raise ValueError("QMI real parts must have the same length as knots")
            if self.imaginary_parts is None or len(self.imaginary_parts) != len(
                self.knots
            ):
                raise ValueError(
                    "QMI imaginary parts must have the same length as knots"
                )
        knots = tuple(float(value) for value in self.knots)
        if any(
            right <= left for left, right in zip(knots[:-1], knots[1:], strict=True)
        ):
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

    @property
    def parameterization(self) -> str:
        return "cartesian" if self.real_parts is not None else "polar"

    @cached_property
    def _cubic_inverse(self):
        """Inverse of the fixed natural-spline system for the QMI knots."""

        if self.interpolation != "cubic" or len(self.knots) <= 2:
            return None
        xp = np.asarray(self.knots, dtype=np.float64) ** 2
        h = xp[1:] - xp[:-1]
        diag = 2.0 * (h[:-1] + h[1:])
        matrix = np.diag(diag)
        if diag.size > 1:
            off = h[1:-1]
            matrix = matrix + np.diag(off, 1) + np.diag(off, -1)
        return jnp.asarray(np.linalg.inv(matrix))

    def _interpolate(self, s, knot_s, values, prepared_index=None):
        if self.interpolation == "linear":
            index, fraction = _interval_index_and_fraction(s, knot_s, prepared_index)
            return values[index] + fraction * (values[index + 1] - values[index])
        if prepared_index is None:
            return _natural_cubic_spline(
                s,
                knot_s,
                values,
                inverse=self._cubic_inverse,
            )

        index, fraction = _interval_index_and_fraction(s, knot_s, prepared_index)
        n = knot_s.shape[0]
        h = knot_s[1:] - knot_s[:-1]
        interior = n - 2
        if interior > 0:
            rhs = 6.0 * (
                (values[2:] - values[1:-1]) / h[1:]
                - (values[1:-1] - values[:-2]) / h[:-1]
            )
            second_inner = jnp.asarray(self._cubic_inverse, dtype=values.dtype) @ rhs
            second = jnp.concatenate(
                (
                    jnp.zeros(1, dtype=values.dtype),
                    second_inner,
                    jnp.zeros(1, dtype=values.dtype),
                )
            )
        else:
            second = jnp.zeros(n, dtype=values.dtype)
        width = knot_s[index + 1] - knot_s[index]
        a = 1.0 - fraction
        b = fraction
        return (
            a * values[index]
            + b * values[index + 1]
            + ((a**3 - a) * second[index] + (b**3 - b) * second[index + 1])
            * width**2
            / 6.0
        )

    def _interpolated_pair(
        self,
        mass,
        first_values,
        second_values,
        prepared_index=None,
    ):
        prepared_fraction = None
        prepared_order = None
        prepared_starts = None
        prepared_ends = None
        if isinstance(prepared_index, tuple):
            if len(prepared_index) < 2:
                raise ValueError("prepared QMI data are incomplete")
            prepared_index, prepared_fraction, *extra = prepared_index
            if extra:
                prepared_order, prepared_starts, prepared_ends = extra

        if mass is None:
            if prepared_fraction is None:
                raise ValueError(
                    "prepared QMI evaluation requires interpolation fractions"
                )
            dtype_source = jnp.asarray(prepared_fraction)
            knot_s = jnp.asarray(self.knots, dtype=dtype_source.dtype) ** 2
            s = None
        else:
            mass = jnp.asarray(mass)
            knot_s = jnp.asarray(self.knots, dtype=mass.dtype) ** 2
            s = mass**2

        magnitudes = jnp.asarray(first_values, dtype=knot_s.dtype)
        phases = jnp.asarray(second_values, dtype=knot_s.dtype)

        if self.interpolation == "linear":
            if prepared_fraction is None:
                index, fraction = _interval_index_and_fraction(
                    s, knot_s, prepared_index
                )
            else:
                index = jnp.asarray(prepared_index, dtype=jnp.int32)
                fraction = jnp.asarray(prepared_fraction, dtype=knot_s.dtype)
            magnitude = magnitudes[index] + fraction * (
                magnitudes[index + 1] - magnitudes[index]
            )
            phase = phases[index] + fraction * (phases[index + 1] - phases[index])
            return magnitude, phase

        if prepared_fraction is not None:
            # Cubic interpolation still uses the fixed interval, but the
            # precomputed fraction avoids reconstructing it from the mass.
            index = jnp.asarray(prepared_index, dtype=jnp.int32)
            fraction = jnp.asarray(prepared_fraction, dtype=knot_s.dtype)
            n = knot_s.shape[0]
            h = knot_s[1:] - knot_s[:-1]

            def cubic(values):
                interior = n - 2
                if interior > 0:
                    rhs = 6.0 * (
                        (values[2:] - values[1:-1]) / h[1:]
                        - (values[1:-1] - values[:-2]) / h[:-1]
                    )
                    second_inner = (
                        jnp.asarray(self._cubic_inverse, dtype=values.dtype) @ rhs
                    )
                    second = jnp.concatenate(
                        (
                            jnp.zeros(1, dtype=values.dtype),
                            second_inner,
                            jnp.zeros(1, dtype=values.dtype),
                        )
                    )
                else:
                    second = jnp.zeros(n, dtype=values.dtype)
                width = knot_s[index + 1] - knot_s[index]
                a = 1.0 - fraction
                b = fraction
                return (
                    a * values[index]
                    + b * values[index + 1]
                    + ((a**3 - a) * second[index] + (b**3 - b) * second[index + 1])
                    * width**2
                    / 6.0
                )

            return cubic(magnitudes), cubic(phases)

        return (
            self._interpolate(s, knot_s, magnitudes, prepared_index),
            self._interpolate(s, knot_s, phases, prepared_index),
        )

    def _interpolated_magnitude_phase(self, mass, prepared_index=None):
        if self.parameterization == "polar":
            return self._interpolated_pair(
                mass,
                self.magnitudes,
                self.phases,
                prepared_index,
            )
        real, imaginary = self._interpolated_pair(
            mass,
            self.real_parts,
            self.imaginary_parts,
            prepared_index,
        )
        value = real + 1j * imaginary
        return jnp.abs(value), jnp.angle(value)

    def interpolated_magnitude_phase(self, mass):
        return self._interpolated_magnitude_phase(mass)

    def _interpolated_cartesian(self, mass, prepared_index=None):
        if self.parameterization == "cartesian":
            return self._interpolated_pair(
                mass,
                self.real_parts,
                self.imaginary_parts,
                prepared_index,
            )
        magnitude, phase = self._interpolated_pair(
            mass,
            self.magnitudes,
            self.phases,
            prepared_index,
        )
        return magnitude * jnp.cos(phase), magnitude * jnp.sin(phase)

    def interpolated_cartesian(self, mass):
        return self._interpolated_cartesian(mass)

    @property
    def prepared_mass_is_self_contained(self) -> bool:
        """Prepared QMI data no longer need the resonance-mass array."""

        return True

    def prepare_mass(self, mass, context: ResonanceContext):
        """Cache the fixed knot interval and interpolation fraction."""

        if int(context.spin) != 0:
            raise ValueError("QMI is defined for a scalar S-wave")
        mass = jnp.asarray(mass)
        knot_s = jnp.asarray(self.knots, dtype=mass.dtype) ** 2
        index, fraction = _interval_index_and_fraction(mass**2, knot_s)
        dtype = jnp.int16 if self.size <= 32767 else jnp.int32
        compact_index = index.astype(dtype)

        # The fixed ordering and interval boundaries are used only by the
        # custom reverse-mode rule.  They cost one int32 per event but replace
        # the much more expensive FP64 scatter-add generated by generic AD.
        order = jnp.argsort(index).astype(jnp.int32)
        counts = jnp.bincount(index.astype(jnp.int32), length=self.size - 1)
        ends = jnp.cumsum(counts).astype(jnp.int32)
        starts = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), ends[:-1]))
        return compact_index, fraction, order, starts, ends

    def evaluate_prepared(self, mass, prepared_index, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("QMI is defined for a scalar S-wave")

        if (
            self.interpolation == "cubic"
            and isinstance(prepared_index, tuple)
            and len(prepared_index) >= 5
        ):
            index, fraction, order, starts, ends = prepared_index[:5]
            dtype_source = jnp.asarray(fraction)
            knot_s = jnp.asarray(self.knots, dtype=dtype_source.dtype) ** 2
            inverse = jnp.asarray(self._cubic_inverse, dtype=dtype_source.dtype)

            def cubic(values):
                return _cubic_qmi_prepared(
                    jnp.asarray(values, dtype=dtype_source.dtype),
                    index,
                    fraction,
                    order,
                    starts,
                    ends,
                    knot_s,
                    inverse,
                )

            if self.parameterization == "cartesian":
                real = cubic(self.real_parts)
                imaginary = cubic(self.imaginary_parts)
                return real + 1j * imaginary

            magnitude = cubic(self.magnitudes)
            phase = cubic(self.phases)
            return magnitude * jnp.exp(1j * phase)

        if self.interpolation == "linear" and isinstance(prepared_index, tuple):
            if len(prepared_index) >= 5:
                index, fraction, order, starts, ends = prepared_index[:5]
                dtype_source = jnp.asarray(fraction)
                if self.parameterization == "cartesian":
                    real_parts = jnp.asarray(
                        self.real_parts,
                        dtype=dtype_source.dtype,
                    )
                    imaginary_parts = jnp.asarray(
                        self.imaginary_parts,
                        dtype=dtype_source.dtype,
                    )
                    return _linear_cartesian_qmi_prepared(
                        real_parts,
                        imaginary_parts,
                        index,
                        fraction,
                        order,
                        starts,
                        ends,
                    )
                magnitudes = jnp.asarray(self.magnitudes, dtype=dtype_source.dtype)
                phases = jnp.asarray(self.phases, dtype=dtype_source.dtype)
                return _linear_qmi_prepared(
                    magnitudes,
                    phases,
                    index,
                    fraction,
                    order,
                    starts,
                    ends,
                )

        if self.parameterization == "cartesian":
            real, imaginary = self._interpolated_cartesian(
                mass,
                prepared_index=prepared_index,
            )
            return real + 1j * imaginary
        magnitude, phase = self._interpolated_magnitude_phase(
            mass,
            prepared_index=prepared_index,
        )
        return magnitude * jnp.exp(1j * phase)

    def __call__(self, mass, context: ResonanceContext):
        if int(context.spin) != 0:
            raise ValueError("QMI is defined for a scalar S-wave")
        if self.parameterization == "cartesian":
            real, imaginary = self.interpolated_cartesian(mass)
            return real + 1j * imaginary
        magnitude, phase = self.interpolated_magnitude_phase(mass)
        return magnitude * jnp.exp(1j * phase)


__all__ = ["QMI"]
