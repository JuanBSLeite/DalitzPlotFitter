# QMI interpolation methods review — 2026-09-10

Scope: correctness of every QMI interpolation mode, in both the 1D isobar form
(`dynamics/lineshape/qmi.py`, modes `linear`/`cubic`/`hermite`/`natural`, polar
and Cartesian parameterizations, including the hand-written custom
`jax.custom_vjp` reverse-mode rules used to avoid FP64 GPU scatter-add) and the
2D Dalitz-amplitude form (`dynamics/qmi2d.py`, modes `none`/`linear`/`cubic`,
including `active_mask` ghost-cell filling and `folded` coordinate exchange).
This builds on the `docs/reviews/20260909_qmi.md` and `20260909_qmi2d.md`
reviews (whose applied fixes remain in place; neither file has otherwise
changed since). Method: read the math, re-derive the custom-VJP formulas by
hand, then reproduce every mode numerically against independent references
(`scipy.interpolate.CubicSpline`, a from-scratch reference bilinear/bicubic
implementation, plain-autodiff versions of the same forward formula, and
central finite differences).

## Main result

All four 1D modes and all three 2D modes reproduce their documented math
exactly: `natural` matches `scipy`'s natural cubic spline to 1e-15; `linear`,
`cubic`, and `hermite`'s custom-VJP gradients match both a plain-autodiff
version of the identical forward formula (to ~1e-14) and central finite
differences (to ~1e-8, consistent with FD truncation) in every configuration
tested, including polar and Cartesian parameterizations and an event landing
exactly on an interior knot; `hermite`'s derivative continuity at interior
knots (the property the mode exists for) holds numerically. QMI2D's bicubic
matches a fully independent from-scratch reference implementation to 2e-16,
reproduces bin-center values exactly, and is numerically C1 across interior
cell boundaries on a nonuniform grid (confirming the fix recorded in
`20260909_qmi2d.md` finding 3 still holds). One real gap was found and fixed:
`QMI2D(folded=True, ...)` accepted `s12_edges != s13_edges` silently, which
distorts the field near the narrower grid's boundary instead of raising.

## 1. Medium: `QMI2D(folded=True, ...)` did not require `s12_edges == s13_edges`

`src/dalitzplotfitter/dynamics/qmi2d.py`, `_coordinates()` (around line 251-253)
and `__post_init__` (no prior check).

`folded=True` computes `s_low = min(s12, s13)`, `s_high = max(s12, s13)`, then
looks up `s_low` on the `s12` grid (`_x_edges_fixed`/`_x_centers_fixed`, built
from `s12_edges`) and `s_high` on the `s13` grid (`_y_edges_fixed`/
`_y_centers_fixed`, built from `s13_edges`). This is only a well-defined,
single symmetric field when both grids represent the same binning — exactly
the documented use case (`docs/lineshapes.md`: "For channels with two
identical particles..."), whose example always passes the identical `edges`
tuple to both `s12_edges` and `s13_edges`. Nothing enforced this: if the two
grids differ, `interpolated_magnitude_phase` clips whichever physical value
ends up as `s_low`/`s_high` into the corresponding grid's own range
(`xx, yy = jnp.clip(x, xc[0], xc[-1]), jnp.clip(y, yc[0], yc[-1])`), silently
distorting the interpolated value for any point beyond the narrower grid's
range instead of raising.

### Reproduction

`s12_edges` on `[0, 1]` (5 bins), `s13_edges` on `[0, 5]` (5 bins), `linear`
interpolation, random magnitudes. Evaluating at `s12=4.2, s13=1.05` (so
`s_low=1.05`, well inside the `s13` grid's range but outside the `s12` grid's
`[0,1]` range) against a "correctly constructed" instance using the same
(wide) edges on both axes, as the documented usage requires:

| Construction | Magnitude at (s12=4.2, s13=1.05) |
|---|---:|
| `s12_edges=[0,1]`, `s13_edges=[0,5]` (mismatched, silently clamped) | 0.595397 |
| `s12_edges=[0,5]`, `s13_edges=[0,5]` (matching, as documented) | 1.591567 |

63% relative difference, no error raised in either case.

### Applied fix

- `QMI2D.__post_init__` now raises `ValueError` when `folded=True` and
  `s12_edges != s13_edges` (exact tuple-of-floats comparison), naming the
  mechanism in the message so a future reader lands on `_coordinates()`
  directly.
- `docs/lineshapes.md`'s folded section gained one sentence stating the
  requirement and why.
- `tests/test_qmi2d.py::test_qmi2d_folded_rejects_mismatched_axis_edges`
  reproduces the rejection.

This does not affect any existing caller: every current use of `folded=True`
in the test suite and the documented example already passes identical edges
on both axes.

## Everything else checked came back clean

### 1D QMI (`dynamics/lineshape/qmi.py`)

- `natural` (global natural cubic spline): the tridiagonal system
  (`_natural_cubic`, diagonal `2*(h[:-1]+h[1:])`, off-diagonals `h[1:-1]`, RHS
  `6*diff(diff(values)/h)`) and the reconstruction formula are the textbook
  natural-spline formulas; matched `scipy.interpolate.CubicSpline(...,
  bc_type="natural")` to 5e-15 max absolute difference over 100 random
  evaluation points.
- `linear`/`cubic`/`hermite` prepared (`evaluate_prepared`, custom-VJP path)
  vs. direct (`__call__`, plain-autodiff path) forward values matched exactly
  (0.0 max difference — same formula, as expected) across both
  parameterizations.
- Custom-VJP gradients (`_linear_qmi_prepared_bwd`,
  `_linear_cartesian_qmi_prepared_bwd`, `_cubic_qmi_prepared_bwd`,
  `_hermite_qmi_prepared_bwd`) matched `jax.grad` through the plain-autodiff
  direct path to ~1e-14 and central finite differences to ~1e-8, for both
  parameterizations, including a test point placed exactly on an interior
  knot boundary (the grouping/interval-assignment edge case). The Hermite
  backward pass was also re-derived by hand from the forward formula
  (`h00*y_i + h10*w*s_i + h01*y_{i+1} + h11*w*s_{i+1}` with `s_i` a
  finite-difference slope depending on up to two neighboring knots) and the
  endpoint/interior chain-rule terms in `_hermite_qmi_prepared_bwd` (lines
  516-542) match the derivation exactly, including the `n_knots == 2`
  degenerate case where both slopes coincide and their gradients must add.
- Hermite interpolation is numerically C1 at an interior knot (left/right
  finite-difference derivatives agreed to 3e-5 at `h=1e-6`, i.e. to FD
  truncation precision) — the property distinguishing it from `cubic`
  (deliberately zero-slope, C1 but not smooth) and `natural` (C2).

### 2D QMI (`dynamics/qmi2d.py`)

- `_bilinear` matched a from-scratch reference bilinear implementation to
  4e-16.
- `_bicubic_one`/`_cubic_axis` matched a from-scratch reference bicubic
  (separable cubic-Hermite with physical-distance central-difference slopes
  and mirrored-ghost-value boundary handling, written independently from the
  source) to 2e-16, and reproduced exact bin-center values.
  `_cubic_axis`'s own internal boundary-mirroring branch (`jnp.where(left==0,
  ...)`/`jnp.where(right==centers.shape[0]-1, ...)`) is unreachable given its
  only call site always passes fixed local indices `left=1, right=2` into a
  4-element patch array — not a bug (`_bicubic_one` performs the actual
  boundary handling externally, consistently), just dead code not touched in
  this pass.
- `none` mode reproduces exact per-bin values at bin centers and correctly
  zeroes intensity/phase for `active_mask`-inactive queries.
- Gradients (plain JAX autodiff, no custom VJP in this file) matched central
  finite differences for `none`/`linear`/`cubic`, and stayed finite for a
  `cubic` field with `active_mask`-inactive corner cells (ghost-filled from
  the nearest active cell) — no NaN-through-zero-branch gradient artifact.
- Bicubic derivative continuity across an interior cell boundary on a
  nonuniform grid held to 2e-5 at `h=1e-6` (FD truncation), confirming the
  `20260909_qmi2d.md` finding-3 fix (physical-distance Hermite slopes
  replacing the old uniform-grid-only Catmull-Rom) still holds.

## Validation

- New/changed: `src/dalitzplotfitter/dynamics/qmi2d.py`, `docs/lineshapes.md`,
  `tests/test_qmi2d.py` (1 new test).
- `pytest tests/test_qmi2d.py tests/test_qmi.py tests/test_qmi_regressions.py`:
  52 passed (was 51 before the new regression test).
- `ruff check` on `qmi2d.py`/`test_qmi2d.py` reports the same pre-existing
  error count as before this change (line-number shifts only) — no new lint
  introduced.
- Full `pytest` suite run for final confirmation (see session notes).

## Limits

This pass re-verified interpolation math and gradients; it did not re-audit
`physical_bin_mask`'s sampled-boundary search or the QMI/QMI2D-to-cache wiring
(`amplitude/cache.py`'s dynamic-recompute path), both already covered by
`20260909_qmi.md`/`20260909_qmi2d.md` and by the per-component normalization
pass in `20260910_iminuit_and_normalization_audit.md`.
