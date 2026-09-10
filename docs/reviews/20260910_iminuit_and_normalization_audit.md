# Full-pipeline audit: integration, iminuit connection, per-component normalization — 2026-09-10

Scope: a full-codebase pass over the three areas most exposed to silent numerical
error — the integration/normalization machinery (`integration/*.py`,
`kinematics/{dalitz_grid,square_dalitz,phase_space_mc,sample}.py`, the
normalization paths in `decay.py`), the JAX-to-iminuit connection
(`fit/minimizer.py`, `fit/parameters.py`, `likelihood/*.py`), and per-component
amplitude normalization (`dynamics/resonance.py`, `amplitude/{components,cache}.py`,
`dynamics/qmi2d.py`, `observables/fractions.py`). Method: read each area, form a
specific hypothesis, and reproduce it numerically (`enable_x64()`, small models
built through the real public classes) rather than conclude from reading alone.

## Main result

The `mean(weights * f)` integration invariant, the square-Dalitz Jacobian, the
narrow-resonance adaptive-grid refinement (including the overlapping-window and
diagonal cases), the Hermitian normalization matrix, per-component unit-integral
rescaling, and the JAX gradients handed to iminuit (dynamics parameters, complex
coefficients, mixture fractions, CP joint coefficients) all reproduced correctly
against independent references, including at points reached mid-`migrad()`. One
real bug was found and fixed: a silent-zero-gradient hazard when the low-level
`PreparedAmplitudeCache` and `Minimizer` are assembled with two different
`Parameter` lists for the same dynamics parameter.

## 1. Medium: `PreparedAmplitudeCache` and `Minimizer` can silently disagree on which DYNAMICS parameters are floating

`src/dalitzplotfitter/amplitude/cache.py`, `PreparedAmplitudeCache.prepare()`
(`floating_owners`/`fixed_indices`/`dynamic_indices`, around line 514) and
`floating_dynamic_owners` (around line 659); `src/dalitzplotfitter/fit/minimizer.py`,
`Minimizer.__init__`/`_backend()`.

`prepare()` decides once, from the `Parameter` sequence it is given, which
components are folded into the compact fixed-evaluation path (`function(data, None)`,
baked into `data_components`/`normalization_matrix_fixed` as plain arrays) versus
which stay in the per-call dynamic-recompute path that reads `fit_values`. This
decision is keyed on `self.parameters`, frozen at prepare time. `Minimizer` is an
independent, decoupled class (by design — it only consumes an opaque
`objective: Callable`) that takes its *own* `Parameter` sequence to decide which
names Minuit steps as free. Nothing cross-checks the two lists agree.

`FitSession`/`CPFitSession`/`DecayModel.prepare_cache()` always derive both from the
same `model.parameters` object, so this cannot happen through the documented
high-level workflow (`docs/fitting.md`'s `Minimizer(nll, model.parameters, ...)`
pattern). It is reachable through the low-level path that
`docs/user_friendly_api.md` explicitly endorses for advanced/validation work: build
`PreparedAmplitudeCache.prepare(..., parameters=X)` directly, then
`Minimizer(objective, Y)` with `Y` marking the same-named dynamics parameter as
floating when `X` had it fixed — e.g. after deciding mid-exploration to float a
parameter without paying to rebuild the (potentially expensive) cache.

### Reproduction

`rho.mass`/`rho.width` fixed at cache-prepare time (`cache.is_compact == True`,
`cache.floating_dynamic_owners == frozenset()`), then a `Minimizer` built with a
`Parameter` list advertising both as floating with bounds:

- `grad(*point)` → `{'rho.mass': 0.0, 'rho.width': 0.0, 'rho.x': 1501.13, 'rho.y': -546.94, 'f0.x': -1724.95, 'f0.y': 2128.85}`
- Direct finite difference on the objective: `NLL(width+1e-4) == NLL(width-1e-4) == NLL(width) == 5159.002420843553` exactly.

The gradient along the mismatched direction is a structural zero, not merely a
small one — Minuit sees a flat direction and reports no error. `evaluate()`
returns identical intensities regardless of the "floating" value (see the added
regression test below).

### Applied fix

- `PreparedAmplitudeCache.check_parameters(parameters)` (`amplitude/cache.py`)
  compares the DYNAMICS floating/fixed split implied by a given `Parameter`
  sequence against `self.floating_dynamic_owners` and raises a `ValueError`
  naming the disagreeing components if they differ.
- `PreparedAmplitudeCache`'s class docstring and `prepare()`'s call site now state
  the hazard explicitly and point advanced callers at `check_parameters`.
- `docs/performance.md` gained a "Hazard: mismatched parameter lists between cache
  and `Minimizer`" note under "JAX and iminuit", explaining why the documented
  high-level workflow is immune and what the low-level contract requires.
- `tests/test_amplitude_cache.py` gained
  `test_check_parameters_accepts_the_list_the_cache_was_prepared_with` and
  `test_check_parameters_rejects_a_dynamics_parameter_fixed_at_a_different_status`;
  the latter also asserts the underlying hazard (`cache.evaluate` returning
  identical intensities for two different "floating" values of the mismatched
  parameter) so the regression stays tied to the actual failure mode, not just the
  new guard.

This is a guard, not a behavior change for any code path that already reuses one
`Parameter` list (i.e. every documented workflow) — `check_parameters` is opt-in
and not called automatically from `Minimizer`, since `Minimizer` deliberately does
not know about `PreparedAmplitudeCache` (see CLAUDE.md's "Two API layers, by
design").

## Everything else checked came back clean

### Integration / normalization (grid quadrature, MC, per-component)

- `mean(weights*f)` reproduced the true Dalitz-plot area for both
  `DalitzGaussLegendreGrid` and `SquareDalitzGrid` against a 20M-event
  `PhaseSpaceMC` reference to ~7e-5 relative.
- The square-Dalitz Jacobian in `kinematics/square_dalitz.py` matches
  `docs/square_dalitz.md`'s stated form exactly, algebraically.
- Narrow-resonance auto-refinement: diagonal (`m12`) case matched a 4000-resolution
  brute-force Square-Dalitz reference to 7e-13 relative; non-diagonal
  (`m13`/`m23`), including two overlapping narrow windows, produced a
  provably non-overlapping segment partition (`adaptive_gauss_legendre.py`'s
  `_axis_segments`) and converged consistently across `binning_factor` values with
  no double-counting.
- A "crossed" narrow resonance (aligned to the axis not being refined) agreed with
  a reference grid explicitly refined for it to 1.2e-6 relative at default
  resolution; the only issue found here is a minor logging asymmetry (the
  gauss-legendre-method branch in `decay.py` doesn't print the same
  "crossed narrow band" notice the square-dalitz-method branch does) — not a
  numeric bug, not fixed in this pass.
- Normalization-matrix construction (chunked compact path and floating-dynamics
  block-update path) reproduces `mean(weights*|sum c_i F_i|^2)` to 2e-13; existing
  tests independently confirm this.
- Per-component normalization: `normalize_component=None` inheritance, Hermitian
  M_ij with interference retained for `normalize_component=False` components,
  value-path/matrix-path rescaling consistency, and — the sharpest check — a
  partial dynamic recompute (including a `QMI2D`-backed `DalitzAmplitude` with two
  floating parameters) matched a from-scratch `PreparedAmplitudeCache` built at the
  same point to 1e-10, for both the cache API and `DecayModel.amplitude()`'s
  independently-coded evaluation path.
- `observables/fractions.py`'s fit-fraction formula reproduced by hand from
  `cache.coefficient_vector`/`cache.normalization_matrix` exactly.
- Two pieces of dead code were noted but not touched (out of scope for a
  correctness-only pass): `amplitude/cache.py`'s `_component_scales` (the
  positive-diagonal-checked variant) is never called, only
  `_component_scales_unchecked` is.

### JAX ↔ iminuit gradient/parameter wiring (beyond Finding 1)

- Gradients matched central finite differences (to ~1e-4–1e-7, consistent with FD
  step-size error) for dynamics parameters, `RealImag` complex coefficients,
  mixture fractions (`MultiBackgroundNLL`), and `CPJointNLL` joint coefficients, at
  the start point, a random point, and a point reached mid-`migrad()`.
- Parameter mapping is name-keyed (`dict`) throughout, not positional — no
  ordering hazard.
- `Minimizer._backend()` structurally excludes `fixed=True` parameters from the
  Minuit vector and closes over their value as a constant, rather than exposing
  them with a masked gradient — this avoids the "zero-gradient but still stepped"
  failure mode by construction for its *own* parameter list (Finding 1 is about
  disagreement between two separately-constructed lists, not this mechanism).
  Bounds are applied per free parameter via `minuit.limits[...]`.
- `evaluate`/`amplitude`/`normalization`/`normalization_matrix` are pure functions
  of the incoming `fit_values` mapping — no host-side mutable memoization keyed on
  parameter values — so there is no stale-cache risk across Minuit steps once
  cache and `Minimizer` parameter lists agree.
- `CPJointNLL`'s shared `I_+ + I_-` normalization verified end-to-end into the
  compiled Minuit objective (`charge_probabilities` summing to 1, correct gradient
  for a CP-asymmetric two-charge model with a shared floating dynamics parameter).

## Validation

- New/changed: `tests/test_amplitude_cache.py` (13 tests, including the 2 new
  ones), `docs/performance.md`, `src/dalitzplotfitter/amplitude/cache.py`.
- `pytest tests/test_amplitude_cache.py tests/test_dynamic_fit_consistency.py
  tests/test_qmi2d.py tests/test_qmi_regressions.py tests/test_minimizer.py`: 58
  passed.
- `ruff check` on the changed files reports the same pre-existing error count as
  before the change (13 in `cache.py`, 1 in `tests/test_amplitude_cache.py`, both
  present before this session's edits) — no new lint introduced.
- Full `pytest` suite run for final confirmation (see session notes; not
  reproduced here since it is the standing CI command).

## Limits

This pass focused on the three areas the task named and did not re-audit toy
generation, backgrounds/SCF, efficiency/veto, or ROOT I/O (each has its own dated
review under `docs/reviews/` already, or is out of scope here). Finding 1 is
reachable only through direct low-level `PreparedAmplitudeCache`/`Minimizer`
construction with two different parameter lists; it does not affect
`FitSession`/`CPFitSession` users.
