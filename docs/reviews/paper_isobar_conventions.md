# Isobar benchmark: angular orientation and CP asymmetry

The notebook `notebooks/benchmark/paper_isobar_benchmark.ipynb` uses the
Cartesian coefficients in Table XXI of Phys. Rev. D 101, 012006.

## Confirmed corrections

- Eq. (33) defines quasi-two-body ACP from squared coefficient magnitudes,
  not independently normalized fit fractions. With the printed coefficients,
  rho ACP is 0.599995% and omega ACP is -3.494441%. The printed values
  have limited precision; exact agreement with Table XVIII is not established.
- Laura++ 3.8 `LauKinematics::calcHelicities` defines theta13 using tracks
  3 and 2 in the 1,3 rest frame, and theta23 using tracks 3 and 1 in the
  2,3 rest frame. The notebook therefore uses ordered pair `(2, 0)`.
  The previous `(0, 2)` reverses the odd-spin factors relative to even spin.
  The generic Zemach implementation is unchanged.
- `make_model(charge, resolution)` now passes resolution to the integrator.
  Previously its argument was ignored and the actual resolution was 1000.

## Remaining discrepancy

At resolution 1000, after the orientation correction, the combined rho-omega
fractions are 61.5598% (B+) and 55.3918% (B-), versus 57.9% and 53.3% in
Tables XII and XIII. The corrected B- mixed-component interference with f2
is -1.3958% (paper -1.4%), and with sigma is +5.7672% (paper +5.7%).
The discrepancy is not fully resolved by the angular correction. In particular,
mixing normalization and the rescattering interference still require validation.
These results do not constitute certification of the fitter or a toy closure test.

Correction to the initial convergence claim: normalization_resolution did not
change the nonadaptive Gauss-Legendre grid. That comparison was not a convergence
test. The notebook now passes explicit normalization_order_m13/m23.

## Rescattering convention audit

Laura++ 3.8 LauRescatteringRes.cc uses an overall factor i and source denominators
1+s/lambda**2, unlike the literal printed expression with m. The new selectable
convention='laura' reproduces these features while retaining the explicit mass
window. The original Laura++ support starts at the KK threshold, whereas the
notebook retains the paper's 1.0--1.5 GeV window. The default library convention
is unchanged; the benchmark explicitly selects 'laura'.

With this convention and actual Gauss-Legendre orders 1400, the combined fractions
are 59.89343% and 54.05918%. At orders 2000 they are 59.89148% and 54.05584%.
Thus the observed change with order is much smaller than the remaining discrepancy.
The earlier orientation-only numbers above are an intermediate diagnostic result.

The official supplementary description identifies Fit3piIsobar.cc as the complete
configuration. Attempts to retrieve it from CDS encountered an automatic browser
verification page; APS returned HTTP 403. The arXiv source archive at
https://arxiv.org/src/1909.05212 contained correlation JSONs but no Isobar C++
configuration. Original mixing normalization and full-precision coefficients
remain unverified; local Laura++ 3.8 is not proof of the analysis's 3.5 setup.

## Local Laura++ 3.5 comparison

The user supplied `/home/juan-leite/Work/Laura-3.5.0/Laura++`.
Comparing function bodies after removing comments and whitespace confirms:

- `LauRhoOmegaMix::amplitude` is identical to local 3.8.
- `LauRescatteringRes::amplitude` is identical to local 3.8, including the
  production factor in s and the overall i.
- `LauSigmaRes::resAmp` and `LauKinematics::calcHelicities` are identical.
- The 3.5 catalog has f2 mass/width 1.2751/0.1851 GeV (3.8: 1.2755/0.1867).
  This is a catalog comparison, not evidence of the analysis configuration.
- `LauIsobarDynamics::calcDPNormalisationScheme` explicitly detects the hidden
  omega mass/width in a rho-omega component for narrow-band integration.
- `calcExtraInfo`, when calculateRhoOmegaFitFractions is enabled, switches
  `setWhichAmpSq(1/2)` and recomputes integrals to extract individual omega/rho
  fractions. This must not be assumed equivalent to independently normalizing
  the two effective terms in the benchmark.

The current user-edited notebook has f2 mass/width 1.256/0.1867 GeV; these
values were preserved. The original Fit3piIsobar.cc is not present in this
installation (GenFit3pi.cc is an example and is not the analysis configuration).
