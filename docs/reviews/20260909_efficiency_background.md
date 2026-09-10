# Efficiency and background review — 2026-09-09

Historical review of functional/histogram models, SignalPDF, FitSession, MultiBackgroundNLL, CP categories and toy compatibility. Findings refer to the original code; applied corrections are listed below.

## Confirmed findings

### 1. High: non-CP mixtures accepted unphysical parameters

MultiBackgroundNLL checked argument combinations, not resolved domains. With two observations and unit densities: f_sig=1.5 gave NLL=0 even under JIT; explicit background fractions 0.8,0.8 gave weights [0.8,0.8,-0.6] and NLL=0; N_sig=3, N_bg=-1 gave finite extended NLL=0.6137056388801094. Individual [0,1] bounds do not constrain a multi-background simplex.

### 2. High: events outside support received finite NLL

Floors in the cached signal log-PDF, SignalPDF and mixture objective turned physical zeros into positive probabilities. A signal-only session with ten vetoed events returned NLL=6907.755278982138 in float64. The veto matched the data s12 values while leaving normalization nodes accepted, isolating event support from a zero integral. In a mixture, signal zero is allowed if a valid background supplies positive total density.

### 3. Medium: non-CP acceptance allowed invalid values and broadcasting

For three events, a (3,1) efficiency produced (3,3) acceptance. A raw callable returning [-1,-1,-1] also passed the helper. This establishes missing validation before broadcasting, not a successful completed fit with an (N,N) array. Histogram constructors already checked edges/values and FunctionalEfficiency mapped invalid values to NaN; arbitrary callables bypassed those protections. Relative efficiencies need not be bounded by one.

### 4. Medium: one-charge CP backgrounds were rejected

CPBackgroundCategory required each charge integral to be strictly positive although its denominator is their sum. plus_values=[1,1], minus_values=[0,0], J_plus=1, J_minus=0 raised a positive-normalization error. Corrected CP toys could generate that case but the category could not represent it.

## Controls and conventions

For NR in D+ -> pi- pi+ pi+ with efficiency 0.3+0.2*s12, the weighted PDF integral was 1.0. Multiplying efficiency by seven changed event density by at most 8.33e-17; cached/direct PDFs differed by at most 5.56e-17. A constant background normalization was 5.063079718469924, matching the mean integration weight.

Signal efficiency is not automatically applied to observed background shapes; veto is applied when apply_veto=True. This avoids double acceptance correction for data-derived templates. Precomputed categories must already match the selection. Histogram values are bin heights, not automatically counts divided by bin area.

## Applied corrections and validation

MultiBackgroundNLL now validates initial and resolved fractions/yields, the simplex and finite total yield. Invalid parameter branches return infinity under JAX. Zero/negative/nonfinite total densities are not floored. SignalPDF returns exact zero outside support and its log-PDF returns minus infinity; the legacy floor argument is retained without clipping densities. FitSession validates scalar/vector acceptance shape, finiteness and sign before multiplication; SignalPDF uses JAX-compatible numerical checks.

CPBackgroundCategory allows a zero integral for one charge only with zero supplied values there; its joint integral must be positive and finite. Non-CP categories still require positive normalization. See `docs/backgrounds_and_vetoes.md`.

Initial validation: **34 passed in 12.45 s** in `test_multi_background`, `test_cp_multi_background`, `test_veto`, `test_workflow`, `test_cp_workflow`, `test_cp_validation`. Reproductions ran separately.

Final validation: **59 passed in 16.40 s**, adding `test_efficiency_background_regressions`. Tests cover JIT/gradients, an analytic valid-point gradient, float32 densities, invalid acceptance shapes, zero support and a one-charge category inside CPJointNLL. Ruff passed for mixture, signal, categories and the new tests; `git diff --check` passed. No statistical efficiency/background closure campaign was run.
