# Laura++ CP fit 415489: histogram convention audit

Audit date: 2026-09-14. Notebook:
`notebooks/data_analyses/22_b2pipipi_cpvfit_laura_comparison.ipynb`.

## Finding

The dominant discrepancy was the background density measure in the notebook.
Laura's input histogram is a density in Square-Dalitz coordinates; it must be
divided by the coordinate Jacobian before mixing with the signal density in
`ds13 ds23`. The notebook instead treated the histogram values directly as a
density in the invariant-mass plane. Automatic total normalization cannot
correct this position-dependent distortion.

Two further mismatches were confirmed: interpolation was disabled by the
choice of Python histogram classes, and the displayed component ACP used a
different definition from Laura's.

## Numerical reproduction

All diagnostic fits use the same 95,110 events (43,723 B+, 51,387 B-), the
same fixed signal/background yields and 43 floating parameters. Each starts
at the printed final Laura parameters, including its masses and widths.
MIGRAD strategy 1, tolerance `1e-4`, float64, CPU; HESSE was omitted because
the experiment measures movement of the minimum, not parameter uncertainties.

The original column is the notebook's saved result. Both diagnostic refits
below used its actual saved resolution, **300 x 300**.

| Parameter | Original notebook | Jacobian + post-veto background normalization | Also interpolate ACC/background | Laura log |
|---|---:|---:|---:|---:|
| rho mass [GeV] | 0.770760 | 0.770472 | 0.770387 | 0.770229 |
| rho width [GeV] | 0.159572 | 0.148847 | 0.148899 | 0.148864 |
| sigma x | -0.309378 | -0.401727 | -0.402839 | -0.402003 |
| sigma mass [GeV] | 0.453811 | 0.505224 | 0.509509 | 0.509084 |
| sigma width [GeV] | 0.357589 | 0.436668 | 0.426424 | 0.425813 |
| f0(1370) mass [GeV] | 1.469310 | 1.507135 | 1.502530 | 1.502180 |

All 43 floating parameters of the fully corrected 300 x 300 fit lie within
0.517 times the corresponding printed Laura error. Before the correction,
the largest difference was 10.672 times that error (`sigma0.x`). These are
descriptive scales, **not calibrated statistical pulls**: the fits use the
same events, so their errors are correlated.

Physical fit fractions also recover the Laura result. For example, after
both corrections at 300 x 300, sigma has 28.11423% (B-) and 21.54693% (B+),
compared with 28.05820% and 21.49981% in Laura. The corresponding original
notebook fractions were 22.4149% and 18.1162%.

Both diagnostic fits converged. Within each objective, the improvement from
the printed Laura point was:

| Objective | NLL at Laura point | NLL at refitted minimum | Change |
|---|---:|---:|---:|
| Jacobian + post-veto normalization, 300 | -491360.555703 | -491362.145069 | -1.589366 |
| Also interpolate, 300 | -491638.768843 | -491639.145684 | -0.376841 |

The original saved minimum was -490595.452076. The external Laura log gives
-491548.7014. Comparing the absolute NLL between different density models
does not isolate a cause; the within-objective changes above do not depend
on additive constants.

## Confirmed source differences

References are the supplied local configuration/log and local Laura++ 3.8
source, not a remote publication:

- `/home/juan-leite/Work/data/fit415489_pipipinotav3_21-5--11h32m13s415489/GenFit3piCP_R2_415489.cc`
- The adjacent `GenFit3piCP_fit_415489.log`.
- `/home/juan-leite/Work/Laura-3.8.0/Laura++/src/`.

1. **Measure:** `LauBkgndDPModel::calcHistValue` explicitly divides the
   interpolated SDP histogram value by `calcSqDPJacobian()`. The package's
   generic `SquareDalitzHistogramBackground` returns bin values. It does not
   infer the measure of an imported histogram.
2. **Interpolation:** the configuration sets `useInterpolation = kTRUE` and
   passes it to ACC and both backgrounds. `Lau2DHistDP::interpolateXY` and
   `Lau2DHistDPPdf::interpolateXY` interpolate between bin centres, retaining
   constant values outside the outer centres. The generic Python histogram
   classes are piecewise constant.
3. **Charge split:** `LauCPFitModel::getTotEvtLikelihood` uses
   `N_bkg * (1 - charge * asym)/2`. The Python session splits a background
   yield using the integrals of its two input shapes. Normalizing the input
   bins in SDP before measure conversion/veto is insufficient. Each converted
   shape now has a unit post-veto integral on the session's quadrature before
   multiplication by `(1 - charge * asym)`. The resulting qqbar B+ probability
   is exactly 0.5039 (checked to `1e-12`). This uses the fitter's integration
   convention `integral(f) = mean(weights * f)`.
4. **ACP:** `LauCartesianCPCoeffSet::acp` calculates
   `-2*(x*dx + y*dy)/(x*x + y*y + dx*dx + dy*dy)`, equivalently the asymmetry
   of squared coefficient magnitudes. The previous notebook calculated the
   asymmetry of independently normalized fit fractions. For its original
   saved rho coefficient, the correct ACP is **-0.0611452%**, whereas its
   old table displayed **-7.65349%**. Correcting that table does not itself
   alter the fit.

## Independent numerical checks

A small C++ bridge was compiled against the installed
`build/lib/libLaura++.so.3.8.0`, constructing the nine resonances through
`LauResonanceMaker` with pair integer 2, radius 4, and the notebook masses and
widths. Each amplitude was symmetrized by evaluating both `(s13,s23)` and
`(s23,s13)`.

At 2,500 points of a 50 x 50 Square-Dalitz grid, with **identical parent and
daughter masses on both sides**, every Python/Laura amplitude ratio was a
positive real constant per component. Maximum relative variation was below
`4e-13`. Scalars, including `SigmaPole`, had ratio one. Spin-dependent
constant factors cancel with per-component unit-integral normalization.
This test rules out an event-dependent barrier/angular/lineshape discrepancy
for this model at these parameters; it does not certify other lineshapes or
the external binary that originally produced the log.

The new interpolation adapter was also compared directly against
`Lau2DHistDP::interpolateXY` on 10,000 deterministic random SDP points and a
30 x 30 synthetic histogram, including folding and outer half-bin regions.
Maximum relative difference was `3.38e-12`. Multiplying the converted
background by the forward coordinate Jacobian recovered that same SDP
density to `5.36e-8` relative at the most ill-conditioned boundary point.

Notebook schema/code-cell compilation and Ruff checks on all added Python
files pass. Six targeted tests pass (`test_laura_comparison_helpers.py` and
`test_square_histograms.py`), covering affine interpolation/folding, the
density measure and the post-veto charge split. Coefficient-level ACP at the
printed Laura values agrees with its log within 0.000114 percentage points,
consistent with the printed input precision. No numerical library hot path
was changed.

## Remaining precision checks

- The original saved code/output used 300 x 300, despite prose claiming
  1000 x 1000. At the **same printed Laura parameters**, the corrected
  objective changes from -491638.768843 at 300 to -491593.487435 at 1000:
  the coarse grid is not converged in absolute NLL. The notebook now defaults
  to 1000 and explicitly calls for comparison to a denser grid before claiming
  numerical closure. Narrow omega/chi_c0 bands are not aligned with its
  selected SDP mass axis.
  Only the fixed-parameter evaluation at 1000 completed in this audit; an
  additional full 1000 x 1000 MIGRAD run was interrupted during the longer
  CPU minimization. No converged fit at that resolution is
  reported or implied by the 300 x 300 results.
- `LauIsobarDynamics::fillDataTree` reads `m13Sq` and `m23Sq`;
  `LauKinematics::updateKinematics` reconstructs `m12Sq` from them and the
  catalog masses. The notebook reads all three branches. The local Laura
  runtime has B mass 5.27934 GeV, while this environment's `particle` gives
  5.27941 GeV. The original log references an external ROOT 6.24 installation,
  so its catalog must be verified before choosing an exact replacement.
  The data branches do not exactly satisfy either local mass sum. The prior
  notebook's speculation that the SDP discrepancy was caused by DTF was not
  established and has been removed.
- A remaining absolute-NLL offset has not been explained. It should not be
  attributed to a Poisson/combinatorial constant without calculating that
  constant and checking matched kinematics and normalization.

The large parameter shifts have been reproduced and removed by concrete
histogram corrections. Agreement to publication precision is a separate,
unfinished validation.

## Applied changes and reproduction

The notebook now uses `laura_comparison_helpers.py` for interpolation and
background measure conversion, normalizes each charge after the veto,
calculates coefficient-level ACP, and labels the error-scaled differences
as descriptive diagnostics. Stale output cells were cleared.

The repository-wide histogram classes retain their existing semantics.
The adapter is specific to this analysis's Laura histogram inputs.
The [machine-readable results](laura_cpvfit_415489_results.json) retain the
original saved parameter values/errors, the diagnostic refit values, the
printed Laura reference and SHA-256 hashes of the external inputs.

With the external input files at the notebook's configured paths:

```bash
OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl-audit \
  .venv/bin/python benchmarks/diagnose_laura_comparison.py \
  --variant jacobian --resolution 300 --fit --output /tmp/jacobian.json

OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl-audit \
  .venv/bin/python benchmarks/diagnose_laura_comparison.py \
  --variant corrected --resolution 300 --fit --output /tmp/corrected.json
```

`--variant original` reconstructs the previous histogram treatment;
omitting `--fit` evaluates the objective at the printed Laura point.
The diagnostic runner opens data through a local file object to avoid an
observed Python 3.14/uproot fsspec asynchronous-reader hang. The decoded
branch values are unchanged.
