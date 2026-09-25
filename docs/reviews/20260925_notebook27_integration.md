# Notebook 27 integration versus local Laura++ 3.8

## Conclusion

The continuous normalization prescription agrees: integrate the coherent,
symmetrized intensity over the full physical mass-squared plane, with efficiency
and veto for the likelihood, retaining interference and jointly normalizing the
two charges. The actual quadrature and the current histogram interpolation are
not identical to the supplied Laura configuration. This is not a certification
of reproduction of the QMI reference table or its fitted parameters.

No physics implementation or notebook settings were changed in this audit.

## Source comparison

Local Laura source: `/home/juan-leite/Work/Laura-3.8.0/Laura++/src/`.
Configuration and log: `/home/juan-leite/Work/data/fit415489_pipipinotav3_21-5--11h32m13s415489/`,
files `GenFit3piCP_R2_415489.cc` and `GenFit3piCP_fit_415489.log`.
That configuration is **isobaric**, not verified provenance for the QMI table.

| Item | Comparison |
|---|---|
| Measure | Both use `ds13 ds23`. In masses, `ds13 ds23 = 4 m13 m23 dm13 dm23`. |
| Square-Dalitz Jacobian | Both give `4 p q m12 (pi/2 * delta_m * sin(pi*mprime)) * pi*sin(pi*thetaprime)`. Our use of `(s12,s13)` is equivalent: replacing `s12` by `s23 = constant-s12-s13` has determinant of absolute value one. |
| Weights | Laura sums physical quadrature weights. Our `mean(weights*f)` stores weights multiplied by the number of points, yielding the same sum. For an n-by-n square Gauss grid, our stored weight is `n²*w_i*w_j*J/4`. |
| Interference | Laura stores `F_i conj(F_j)` and contracts with `c_i conj(c_j)`; we store `conj(F_i) F_j` and compute `c† M c`. These are the same real integral. |
| Symmetry | Both integrate the full domain and symmetrize the amplitude before squaring. Folding a histogram lookup does not halve the integration domain. |
| Veto | Both multiply the accepted intensity and its normalization by the same binary mask on both opposite-charge pairs. Only exact endpoint inclusion differs. |
| CP normalization | Laura's `2/(I_minus+I_plus)` is multiplied by `N_signal/2`; our direct denominator is `I_minus+I_plus`. No residual factor of two. |
| Physical fractions | Bare integrals exclude efficiency/veto; accepted likelihood integrals include them. The notebook builds separate physical caches. |
| Component scales | Laura explicitly carries `fNorm_i`; notebook 27 disables individual component normalization. Coefficient comparisons require a base-scale conversion; this is separate from the integration measure. |

Source entry points: Laura `LauDPPartialIntegralInfo.cc:153`,
`LauKinematics.cc:176`, `LauIsobarDynamics.cc:2084,2531`,
`LauCPFitModel.cc:2376,2486`; ours `kinematics/square_dalitz.py`,
`integration/matrix.py`, `cp_workflow.py`, `likelihood/cp.py`.

## Actual algorithm differences

1. **Signal grid.** Notebook 27 requests Square-Dalitz, pair `(0,1)`, resolution
   500. Omega and chi_c0 live in the crossed `(0,2)`/`(1,2)` bands, so the
   automatic refinement does not apply: there are exactly 250,000 nodes.
   The supplied Laura log, starting at line 213, instead records mass-plane
   subdivisions around omega and chi_c0: approximately 999/1000 nodes across
   their windows, and coarse 0.005 GeV target spacing elsewhere. Neither this
   global square grid nor that isobaric partition explicitly follows every
   step-QMI edge. Same integral, different discretization.
2. **Background integration.** Laura `Lau2DAbsHistDPPdf::calcNorm` uses midpoint
   summation with 0.001 axis-range steps. For a Square-Dalitz histogram this
   integrates `H(mprime,thetaprime)*V` directly. Our background evaluation is
   `H/J`, integrated with `J` in the weights on the signal quadrature. The
   Jacobian cancels analytically. This explains why mass-plane quadrature of
   the same background converges poorly near its physical boundary.
3. **Integrand difference in the saved notebook.** Both efficiency and
   backgrounds currently specify `interpolation="spline"`; the Laura config
   uses `setEffHisto`/`setBkgndHisto` with interpolation enabled (bilinear).
   Notebook prose describing bilinear interpolation is therefore stale.
   At fixed parameters and 2000² nodes, replacing only the signal efficiency
   with linear interpolation changes the accepted integral by -0.10605% (B+)
   and -0.07149% (B-). This is not a quadrature effect.
4. **Literal Laura Gauss implementation.** In local `LauIntegrals.cc:63`,
   `zSq` is initialized before Newton iteration and not updated with `z`.
   The printed weight sums in the supplied log are consequently slightly
   below two. A direct numerical transcription gives sum=1.999766576664 for
   order 92, 1.999988657382 for 500, and 1.999996815536 for 1000. NumPy's
   Legendre rule does not reproduce that defect. This source-level numerical
   difference should not be copied into our integration.

## Fixed-parameter numerical check

`benchmarks/audit_notebook27_integration.py` reconstructs the current saved
model and maps using the rounded parameter table printed in cell 35. It executes
setup only, never a fit. The result is conditional on that printed point and
current source: notebook outputs do not prove which source produced an older fit.

| Square Gauss nodes per axis | Accepted integral B+ | Accepted integral B- |
|---:|---:|---:|
| 500 | 54.69256775 | 65.78667312 |
| 1000 | 54.68838041 | 65.77944530 |
| 1500 | 54.68609178 | 65.77563353 |
| 2000 | 54.68730837 | 65.77560704 |

From 500 to 2000, changes are **-0.00962% / -0.01682%**. Individual bare
component integrals move by -0.06670% (omega) and -0.27207% (chi_c0);
these relative changes are the same for both charges because dynamics are fixed.
The bare phase-space area stays 376.612031031 GeV^4 to about 1e-12 relative.
No tested grid node changes acceptance between our inclusive mass comparison
and Laura's strict squared-mass comparison.

An independent tensor Gauss rule in `(m13,m23)`, partitioned at QMI edges,
veto edges and narrow-resonance windows, gives accepted signal integrals
54.68906978 / 65.77729838 at order 96 per cell (6,144,158 retained points).
These differ from square 2000 by +0.00322% / +0.00257%. This diagnostic is
not an execution of Laura's exact grid or its Gauss-weight routine.

For backgrounds, square Gauss 2000 gives raw post-veto integrals
3.43799602 / 3.41874526; square midpoint 1000 gives
3.43802986 / 3.41878896 (same spline maps, not the Laura bilinear inputs).
Square 500 to 2000 changes them by +0.00096% / +0.00198%.
The independent mass-plane rule still differs by about half a percent and
oscillates with order for these `H/J` shapes; it is not a converged background
reference. Do not interpret it as a failure of the square-coordinate measure.

## Validation and remaining scope

Reproduction from repository root:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python benchmarks/audit_notebook27_integration.py \
  --output docs/reviews/20260925_notebook27_integration_results.json
```

`tests/test_square_dalitz.py`, `tests/test_gauss_legendre_integration.py`, and
`tests/test_veto.py`: **20 passed**. New diagnostic script passes Ruff.
Full numeric outputs and rounded parameters are retained in the adjacent JSON.

The observed integral shifts do not measure the change of the best-fit
parameters. A fit-level closure requires identical maps/interpolation, masses,
amplitude conventions and QMI definition, followed by evaluating likelihood
gradients and refitting at increased quadrature resolution. In particular,
the QMI-table generator/configuration is not established by the supplied
isobaric Laura log. No claim is made that integration explains, or rules out,
the table discrepancies.
