# Square-Dalitz integration review — 2026-09-10

Scope: the Square-Dalitz coordinate/Jacobian machinery
(`kinematics/square_dalitz.py`: `square_dalitz_to_invariants`,
`invariants_to_square_dalitz`, `square_dalitz_jacobian`, `SquareDalitzGrid`)
and its adaptive narrow-resonance-refinement counterpart
(`integration/adaptive_square_dalitz.py`: `AdaptiveSquareDalitzGrid`), plus
`decay.py`'s decision logic for selecting between them
(`normalization_sample`, `normalization_scheme`, `_adaptive_narrow_resonances`).
The forward map, Jacobian, and the `mean(weights*f)` convention had already
been spot-checked by one of the three parallel agents in this session's first
review pass (`docs/reviews/20260910_iminuit_and_normalization_audit.md`); this
pass re-derives the Jacobian algebraically from scratch, re-verifies the
weight convention by hand for both grid classes, and adds checks that pass
did not cover: coordinate round-trip, the documented pair-reversal symmetry,
and — where the one real finding of this pass came from — what happens when
`AdaptiveSquareDalitzGrid`'s fixed `max_depth` is insufficient for an
extremely narrow resonance.

## Main result

The forward map, Jacobian, and both grid classes' weight conventions are
correct: re-derived `square_dalitz_jacobian` term-by-term from
`square_dalitz_to_invariants`'s formulas (matches exactly), confirmed the
coordinate round-trip and pair-reversal symmetry to 1e-15, and confirmed by
hand-deriving the `mean(weights*f)` scaling for both `SquareDalitzGrid`
(`0.5*n*w` per axis) and `AdaptiveSquareDalitzGrid` (`raw_weight * N`, N =
total point count) that they encode the identical convention despite
different-looking constructions. A 2 MeV test resonance's adaptive integral
matched a brute-force uniform `SquareDalitzGrid` reference to 4.6e-7 relative
as the reference resolution was refined toward it (2000 → 4000 → 8000
points/axis: 4.4%, 0.099%, 4.6e-5% error, converging cleanly onto the
adaptive value). One real gap was found and fixed: `AdaptiveSquareDalitzGrid`
could silently under-resolve a resonance narrower than
`binning_factor`/`max_depth` can reach, with no error or warning.

## 1. Low-medium: `AdaptiveSquareDalitzGrid` silently under-resolved resonances narrower than `max_depth` could reach

`src/dalitzplotfitter/integration/adaptive_square_dalitz.py`, `_mprime_cells()`
(previously lines 119-140).

The adaptive m' tree stops refining a cell once `depth == max_depth`
(default 12), regardless of whether `_needs_refinement` (`mass_span >
cell_order*pole_width/binning_factor`) is still true there. For most narrow
resonances this never matters — `max_depth=12` gives a 4096x refinement of
the base cell, far more than typical widths need — but for a sufficiently
narrow resonance (in practice, roughly below ~0.1-0.2 MeV for typical
three-body charm-meson kinematics with default settings) the tree exhausts
`max_depth` before satisfying its own target resolution, and returns a grid
that is quietly coarser than intended. Nothing detects or reports this:
`DecayModel` doesn't even expose `max_depth`/`cell_order` as constructor
parameters, so a user hitting this has no visible signal anything is wrong
and no way to fix it short of manually constructing
`AdaptiveSquareDalitzGrid` and injecting its `sample()` via
`DecayModel(normalization_sample=...)`.

### Reproduction

D0-like three-body kinematics (`mother_mass=1.86484`, three pion masses),
`pole_mass=0.7700`, `pole_width=0.00005` (50 keV — an extreme but not
unphysical case, well inside the documented "narrow" bucket of
`normalization_narrow_width<=20 MeV` default), default `binning_factor=100`,
`max_depth=12`: the target per-cell mass span is `4.000e-6` GeV, but the
finest cell actually reached inside the resonance window only achieved
`7.504e-5` GeV — **18.76x coarser than intended**, with the old code
returning this grid with no error, no warning, and no field on the returned
sample indicating it.

### Applied fix

- `_mprime_cells()` now tracks which leaves were forced to stop by the depth
  cap while `_needs_refinement` was still true, and raises `ValueError`
  listing one such cell and pointing at the fix (`max_depth`, `cell_order`,
  `binning_factor`, or the manual-`normalization_sample` escape hatch) instead
  of returning a silently under-resolved grid.
- Verified this does not trigger for any resonance width in the existing test
  suite or in a 20 MeV (the documented narrow-resonance threshold) test case
  with default settings — only the pathological sub-0.1 MeV case.
- `tests/test_gauss_legendre_integration.py` gained
  `test_adaptive_square_dalitz_raises_instead_of_silently_under_resolving`:
  the 50 keV case now raises with the expected message, and the same
  configuration with `max_depth=20` (enough depth for this specific case)
  succeeds and returns a non-empty sample.

## Everything else checked came back clean

- **Jacobian**: re-derived `square_dalitz_jacobian` by hand from
  `square_dalitz_to_invariants`'s `m_ij(mp)` and `s_ik(mp,tp)` formulas —
  `|d(s_ij)/dmp| * |d(s_ik)/dtp|_{fixed mp}` (the map's Jacobian matrix is
  upper-triangular since `s_ij` doesn't depend on `tp`, so the determinant
  is exactly this product) — matches the implementation term-for-term.
- **Coordinate round-trip**: `invariants_to_square_dalitz(square_dalitz_to_invariants(mp,
  tp))` recovers `(mp, tp)` to 6e-15 max absolute error over 500 random
  points in `(0.01, 0.99)^2`.
- **Pair-reversal symmetry** (documented in `square_dalitz_to_invariants`'s
  docstring: "Reversing the pair preserves `mprime` and maps `thetaprime` to
  `1 - thetaprime`"): confirmed `square_dalitz_to_invariants(mp, tp, pair=(0,1))
  == square_dalitz_to_invariants(mp, 1-tp, pair=(1,0))` to 3e-15 over 200
  random points.
- **`mean(weights*f)` convention, both grid classes**: re-derived by hand.
  `SquareDalitzGrid`'s per-axis `stored_weights = 0.5*n*w` (raw
  `numpy.polynomial.legendre.leggauss` weights `w`, scaled by `0.5*n`) makes
  `mean(weight_mp*weight_tp*jacobian*f)` reduce exactly to the standard
  `0.25*sum(w_mp*w_tp*jacobian*f)` 2D Gauss-Legendre-on-`[0,1]^2` quadrature.
  `AdaptiveSquareDalitzGrid` instead builds physically-scaled per-segment
  weights directly (a "plain sum" convention: `sum(weight*f) ≈ integral`) and
  then multiplies by the total point count `N` once at the end
  (`weights = jacobian * raw_weights * float(mp.size)`) to convert to the
  same `mean()`-based convention — algebraically equivalent despite the very
  different-looking construction.
- **Narrow-resonance accuracy** (realistic 2 MeV width, not the pathological
  case above): adaptive integral converged onto a brute-force uniform
  reference as reference resolution increased (2000/4000/8000 points per
  axis: 4.36e-2 → 9.94e-4 → 4.62e-7 relative difference from the adaptive
  value), confirming the adaptive value itself is accurate, not merely
  self-consistent.
- **`decay.py`'s normalization-path selection** (`_adaptive_narrow_resonances`,
  `normalization_sample`, `normalization_scheme`): read through the full
  branch structure (gauss-legendre vs. square-dalitz method, diagonal m12 vs.
  m13/m23 narrow bands, aligned vs. crossed resonances) — logic and informative
  `print("INFO ...")` messages are consistent with the chosen grid class in
  every branch. The only issue here is one already on record from the prior
  integration review: the gauss-legendre-method's diagonal-m12 branch doesn't
  print the same "crossed narrow band kept at standard resolution" notice the
  square-dalitz-method branch does — a logging asymmetry, not a numeric bug,
  left as-is.

## Validation

- New/changed: `src/dalitzplotfitter/integration/adaptive_square_dalitz.py`,
  `tests/test_gauss_legendre_integration.py` (1 new test).
- `pytest tests/test_square_dalitz.py tests/test_integration.py
  tests/test_decay_model.py tests/test_mc_integration_regressions.py
  tests/test_gauss_legendre_integration.py`: 67 passed.
- `ruff check` on the changed file: unchanged (clean before and after).
- Full `pytest` suite run for final confirmation (see session notes).

## Limits

This pass did not re-run the overlapping-narrow-window / crossed-resonance
scenarios already reproduced numerically in the prior integration review, and
did not stress-test `AdaptiveSquareDalitzGrid` against `toy-mc` normalization
or GPU execution.
