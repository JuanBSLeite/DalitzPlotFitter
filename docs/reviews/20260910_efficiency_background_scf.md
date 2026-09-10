# Efficiency and background inclusion review — 2026-09-10

Scope: how efficiency and background are wired into the fit machinery —
`workflow.py`/`cp_workflow.py`'s `_acceptance` helpers, `SignalPDF`
(`pdf/signal.py`), `MultiBackgroundNLL`/`BackgroundCategory`
(`likelihood/mixture.py`, `background/categories.py`), `CPJointNLL`'s
per-charge efficiency and `CPBackgroundCategory`
(`likelihood/cp.py`), and — not covered by the prior
`docs/reviews/20260909_efficiency_background.md` pass — self-cross-feed
(`pdf/scf_signal.py`, `resolution/scf.py`). Builds on that prior review, whose
applied fixes (input validation for efficiency/background models, exact-zero
support outside the physical domain, the non-CP mixture simplex fix, the
one-charge CP background fix) remain in place; none of those files changed
source since. Method: read the wiring, then reproduce suspected discrepancies
numerically (`enable_x64()`, small models built through the real public
classes) rather than conclude from reading alone.

## Main result

`FitSession`/`CPFitSession`'s efficiency*veto product (`_acceptance`) is
applied identically and consistently to both the data numerator and the
normalization integral on both the signal and (for veto only) background
sides, matching the documented convention that efficiency is not
automatically applied to background templates. `MultiBackgroundNLL` and
`CPJointNLL`'s simplex/yield validation, zero-support handling, and joint
charge normalization all reproduced correctly. One real bug was found and
fixed in `SCFSignalPDF.normalization()`: its `veto is None` fast path used a
different, inconsistent formula from the `veto is not None` path, so adding a
physically no-op (always-accepting) veto silently changed the computed
normalization, and the unvetoed path was not actually the integral of its own
`numerator()`.

## 1. Medium: `SCFSignalPDF.normalization()`'s unvetoed shortcut was inconsistent with the vetoed formula and with its own `numerator()`

`src/dalitzplotfitter/pdf/scf_signal.py`, `normalization()` (previously lines
51-71).

The vetoed branch computes the total density as a split: a continuously
integrated correctly-reconstructed (CR) term plus a migration-binned SCF term
built from `SquareDalitzSCFMap.smeared_bin_density`, which approximates each
SCF bin's contribution using `efficiency*intensity` evaluated once at the
bin's centre (a midpoint-rule approximation, consistent with how
`numerator()` builds the per-event SCF density via the same map). The
unvetoed branch instead computed the total as `integrator.integrate(efficiency
* intensity)` — a single continuous integral over the *entire* domain,
skipping the SCF map's binning altogether. `docs/scf.md`'s "Without vetoes"
section asserted these were exactly equal ("Since every true-bin migration
distribution is normalized, `integral rho_reco(r) dr = integral epsilon(t)
|A(t)|^2 dt`"), which only holds when `efficiency*intensity` is constant
within each SCF bin (or in the limit of infinitely fine `bins_mprime`/
`bins_thetaprime`) — not in general.

### Reproduction

D-like three-body decay, `intensity = 1 + 0.3*s12 + 0.1*s13`, a 6x6
Square-Dalitz SCF map with a smeared (non-identity) migration matrix and
random `scf_fraction` per bin, 40-point Gauss-Legendre Square-Dalitz
integration grid:

| Configuration | `normalization()` |
|---|---:|
| `veto=None` (old shortcut) | 7.367409 |
| `veto=` an always-accepting `FunctionalVeto` (physically a no-op) | 7.419214 |

0.70% relative difference between two configurations that should be
physically identical. Independently re-integrating the *actual* `numerator()`
formula on the fine grid gave 7.467502, i.e. `mean(weights * pdf(data))` was
**1.36% away from 1** for `veto=None` and **0.65% away** even with the
trivially-accepting veto — confirming `SCFSignalPDF` was not correctly
unit-normalized relative to its own density formula in either branch, worst
in the unvetoed shortcut.

### Applied fix

- `normalization()` now always uses the CR+SCF split formula (`self._acceptance`
  already returns an all-ones array when `veto is None`, so this is not a
  special case); the single-continuous-integral shortcut is removed.
- `docs/scf.md`'s "Without vetoes" section now states the split identity that
  actually holds unconditionally, and explains when it reduces to the
  continuous integral (bin-wise-constant `efficiency*intensity`, or
  arbitrarily fine SCF binning) rather than claiming exact equality.
- `tests/test_scf.py` gained
  `test_scf_normalization_is_unaffected_by_an_always_accepting_veto` (bit-exact
  equality between `veto=None` and an always-accepting `FunctionalVeto`, which
  was the sharpest reproduction) and `test_scf_pdf_integrates_to_one_without_a_veto`
  (`mean(weights * pdf(data))` within 5e-3 of 1 for a 10x10 SCF map — loose
  enough to tolerate the SCF map's own intended coarse-graining, tight enough
  to catch a regression back to the old shortcut).

After the fix, the remaining small deviation from exact unit normalization
(checked from 6x6 up to 48x48 SCF bins: ~3e-3 down to ~6e-4) is the SCF map's
own intended midpoint-rule coarse-graining — the same kind of discretization
inherent to any histogram-based efficiency/background model in this codebase
— and shrinks with finer `bins_mprime`/`bins_thetaprime`, not a further bug.

## Everything else checked came back clean

- `workflow.py`'s `_acceptance` multiplies efficiency and veto once and
  validates finiteness/non-negativity before either the signal numerator or
  the cache's `efficiency_normalization` uses it, so data-side and
  normalization-side acceptance can never drift apart for the documented
  `FitSession` path. `_build_background` deliberately does not apply
  `efficiency` to background templates (only `veto`, gated by
  `apply_veto`), matching the documented convention that background shapes
  are already selection-matched data-derived templates.
- `cp_workflow.py` mirrors this per-charge (`plus_efficiency`/
  `minus_efficiency`, `plus_veto`/`minus_veto`, independently defaultable),
  and `CPJointNLL._signal_densities` divides both charges' data-side
  intensities by the *same* `integral_plus + integral_minus` (each already
  efficiency-weighted at cache-prepare time), correctly implementing the
  joint-normalization convention CLAUDE.md describes.
- `MultiBackgroundNLL`/`CPJointNLL`'s simplex, yield, and zero-support
  validation (the prior review's applied fixes) still behave correctly:
  spot-checked invalid fraction/yield combinations and confirmed they route
  to `+inf` rather than a finite but unphysical NLL.
- `BackgroundCategory`/`CPBackgroundCategory` validate non-negative, finite
  values and a positive joint normalization at construction, so a background
  density can never go negative or divide by zero downstream.
- `pdf/signal.py`'s `SignalPDF` (the non-SCF, non-cached generic path) has no
  analogous shortcut/split branching — its normalization is always the same
  single continuous integral used for both the numerator and denominator, so
  it does not share the SCF class's bug pattern.

## Validation

- New/changed: `src/dalitzplotfitter/pdf/scf_signal.py`, `docs/scf.md`,
  `tests/test_scf.py` (2 new tests, now 11 total).
- `pytest tests/test_scf.py`: 11 passed.
- `ruff check` on the changed files reports the same pre-existing error count
  as before this change — no new lint introduced.
- Full `pytest` suite run for final confirmation (see session notes).

## Limits

This pass did not re-audit `discriminants.py` or `resolution/convolution.py`
(the discriminating-variable-PDF and 1D-convolution pieces of the pipeline);
neither references efficiency/veto/background directly, so they were judged
out of scope for "efficiency and background inclusion" specifically, but they
compose with this machinery at a higher level and were not exercised here.
