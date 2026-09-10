# Toy generation review — 2026-09-09

Historical review of accept-reject, inverse-transform/Rosenblatt sampling, charge splitting, background mixing and four-momenta. CPU tests used separate processes with two cores. Corrections follow the original findings.

## 1. High: inverse-transform leaked into vetoed regions

Continuous interpolation of tabulated quantiles, including collapsed CDF plateaus, filled zero-density gaps without a final support check. For constant density in D+ -> pi- pi+ pi+, excluding 0.8<=s12<=1.2 GeV², 50000 events with seed 112 gave:

| Resolution | Veto violations |
|---|---:|
| 64 | 816 |
| 256 | 280 |
| 1024 (default) | 44 |

An accept-reject control had zero violations in 2000 events, seed 23, pool_size=4096. Increasing CDF resolution reduced but did not eliminate the issue. Rejecting invalid candidates guarantees support, not exact density within allowed regions.

## 2. High: momentum-dependent accept-reject lost angular selection

The fallback generated full four-momenta but unconditionally discarded accepted momenta. Final reconstruction sampled new orientations. For constant signal with `efficiency=lambda d:d['p1'][:,3]>0`, 2000 events, seed 24, pool_size=4096, include_momenta=True, **1002 events** violated the original angular selection. This does not establish bias in Genfit without directional acceptance.

## 3. Medium: inverse CP generation prepared a zero-rate charge

Both signal samplers were prepared before checking counts. NR with CPRealImag(1,0,1,0), 100 events and seed 1 failed with a zero/invalid target-integral error for inverse-transform; accept-reject returned 100 plus and zero minus.

## Structure and numerical limits

Accept-reject includes proposal weights, monitors envelopes and restarts accumulated samples after envelope updates. Unvisited pilot cells retain positive proposal support. A pilot is not proof of a global envelope for arbitrary unresolved peaks.

Inverse-transform includes `2*m12*(s13_max-s13_min)` but remains a grid approximation. Its CDF resolution and the fit normalization grid require separate convergence checks. CP uses accepted integrals for binomial charge splitting; backgrounds use multinomial counts. Returned toys have unit event weights.

## Applied corrections

Full marginal CDF intervals preserve jumps across plateaus, including endpoint gaps. Candidates are checked against the original density at the returned invariant precision and zero-density candidates are replaced; failure after 100 batches is explicit. Conditional interpolation remains approximate.

Accept-reject retains accepted orientations in its momentum-dependent path. Mixed compact components alone are reconstructed; include_momenta=False removes momenta after selection. Inverse CP prepares only positive-count charges. Both methods validate finite non-negative charge integrals with positive sum and skip absent signal normalization. Prepared generation skips zero-weight signal/background components. Model and callbacks must remain unchanged while reusing prepared CDFs.

Original reproductions after correction: **zero veto violations in 50000 events at each of 64/256/1024**, **zero angular violations in 2000 events**, and **100 plus/zero minus** with either CP method.

## Validation

Initially 16 toy tests passed in 13.53 s (one ROOT test deselected), and two inverse-transform tests passed in 1.33 s with JAX_ENABLE_X64=true. The latter fail in float32 because conservation tolerances are around 1e-10, independently of the reproduced float64 bugs.

The ROOT test blocked even in a separate process. A 60-second traceback showed a wait in uproot's `_ranges_or_baskets_to_arrays` during `tree.arrays(['charge'],library='np')`. It was interrupted and remained unvalidated; that alone does not establish a generation defect.

After corrections: **35 passed, 1 deselected**, covering toy regressions, toy API and inverse-transform tests. Coverage includes reproducibility, disconnected-support population checks against independent integration, public signal/mixture/CP veto paths, angular mixtures, compact output, either zero-rate charge, invalid joint rates and pure backgrounds. Ruff and diff checks passed. No ROOT I/O fix was claimed.
