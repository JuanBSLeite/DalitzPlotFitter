# 1D PDF convolution review — 2026-09-10

Scope: `resolution/convolution.py` (`ConvolvedPDF1D`, `GaussianResolution1D`)
and its interaction with `discriminants.py`'s 1D PDFs, following the same
"reproduce suspected discrepancies numerically" method used for the other
reviews this session. This module had no dedicated prior review; it is
already covered by `tests/test_convolution.py` (6 tests: finite-window
normalization, narrow-resolution limit, bias, parameter resolution, a
`RelativisticBreitWigner`-based closure, and `FactorizedDensity` composition).

## Main result

No bug found. `ConvolvedPDF1D.normalization()` and `_numerator()` implement
consistent halves of the same double integral (swapping integration order
algebraically relates them, unlike the SCF case reviewed earlier this
session, where one branch used a fundamentally different, fixed-resolution
discretization) — both are controlled by the same `quadrature_order` and
converge to the same continuum value as it increases. `GaussianResolution1D`'s
`convolve_pdf` uses a sign convention (`true = observed - bias +
sqrt(2)*sigma*node`) that looks reversed against the standard substitution
derivation, but `numpy.polynomial.hermite.hermgauss`'s nodes/weights are
exactly symmetric about zero, so the sign is immaterial to the resulting
sum — verified this is not a bug by rederiving the substitution by hand and
confirming the existing test suite's tight normalization/gradient checks pass.

## What was verified numerically (beyond the existing test suite)

1. **Generic (non-`convolve_pdf`) numerator path** — untested by the existing
   suite, since every current test uses `GaussianResolution1D` (which always
   takes the specialized Gauss-Hermite path). Built two custom kernels with no
   `convolve_pdf` method — a discontinuous box kernel and a continuous
   (kinked) triangular kernel — with independently-derived closed-form
   `interval_probability` and reference values from `scipy.integrate.quad`.
   Both matched `ConvolvedPDF1D`'s generic Gauss-Legendre fallback to the
   precision expected from *ordinary* (non-spectral) Gauss-Legendre
   quadrature applied to a non-smooth integrand: errors at the 1e-3–1e-4
   level for the box kernel at order 128–2048, and 1e-4–1e-3 for the kinked
   kernel at order 128–256, shrinking as `quadrature_order` increases in both
   cases (checked explicitly: box-kernel error at one representative point
   went 3.8e-3 → 8.5e-4 → 9.8e-5 → 7.4e-5 for order 128/512/2048/8192).
   `GaussianResolution1D`'s own kernel is analytic, so its specialized
   Gauss-Hermite path gets the much faster (spectral) convergence the
   existing tests already exploit at order 128–192; the generic path's slower
   convergence for non-smooth custom kernels is expected numerical behavior,
   not a formula defect — same conclusion either way when cross-checked
   against an independent reference.
2. **Gradient correctness** — not previously tested (existing tests only
   check qualitative parameter ordering). `jax.grad` of `-sum(log(pdf(x,
   {"sigma": s})))` w.r.t. the Gaussian resolution's `sigma` matched a central
   finite difference to 8e-10.
3. **No NaN-gradient hazard near small sigma** — checked `sigma` = 0.001,
   0.01, 0.05 (values a floating resolution parameter could plausibly probe
   near its lower bound during a fit): gradients stayed finite in every case,
   confirming the `jnp.where(valid, raw, 0.0)`/`jnp.where(inside, ...)` masking
   pattern used throughout this file does not leak a NaN gradient through the
   masked-out branch for these kernels.
4. **`np.polynomial.hermite.hermgauss` node symmetry** (the fact the
   `convolve_pdf` sign convention relies on): confirmed nodes/weights are
   exactly antisymmetric/symmetric respectively, so the `+`-sign substitution
   used in the code sums over the identical set of (node, weight) pairs as
   the `-`-sign substitution the direct derivation gives.

## Note (not a bug)

`ConvolvedPDF1D.__post_init__` always builds both the Gauss-Legendre *and*
Gauss-Hermite node/weight arrays at `quadrature_order`, even for a kernel
whose `convolve_pdf` is never used (the generic-path kernels above still pay
for an unused Hermite table). At very high orders (~8192+) this surfaced
`numpy.polynomial.hermite.hermgauss` overflow/`RuntimeWarning`s on this
environment's numpy version — a numpy-side numerical limitation of
`hermgauss` at extreme order, not something this library controls, and far
above the default (`quadrature_order=96`) or any order used in the existing
test suite (≤512). Not fixed in this pass: harmless at realistic orders and
orthogonal to correctness.

## Validation

- No source changes; `pytest tests/test_convolution.py`: 6 passed (unchanged).
- Additional numeric reproductions described above were run standalone, not
  added as new tests, since they exercise a synthetic non-smooth kernel
  chosen only to probe the generic path's formula and are not representative
  of a realistic detector-resolution model the codebase should carry as a
  regression fixture.
