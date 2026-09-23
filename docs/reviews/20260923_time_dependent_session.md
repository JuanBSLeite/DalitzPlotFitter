# Time-dependent session and Dalitz diagnostics audit (2026-09-23)

The review covered `TimeDependentFitSession` and the Dalitz diagnostics added
alongside the low-level time-dependent likelihood. It did not close the
separate Belle amplitude-convention discrepancies recorded in
[the Belle reproduction review](20260923_time_dependent_belle.md).

## Reproduced failures

- A floating rho mass changed from 0.74 to 0.82 GeV left the session NLL at
  116.9241947577607 with zero mass gradient. An independently configured
  amplitude cache changed the amplitude by 41.9% in relative array norm.
- Changing a fitted rho coefficient from 0.2 to 2.0 left the projected bin
  contents unchanged to about 2e-15. A nonuniform Dalitz efficiency also left
  that projection unchanged.
- A component explicitly normalized in A was left raw in reflected Abar.
  A coarse rho+NR test gave integrated intensities 3.813 versus 7.521.
- With x=0.2, y=0.3, tau=0.4103, |q/p|=1.4, phi=0.2 and temporal acceptance
  t/(t+0.3), the old Dalitz marginal differed from direct integration of the
  joint likelihood by up to 4.83%. These are stress-test mixing values, not
  an estimate of the discrepancy at the Belle parameters.
- A snapshot with sample-wide mistag 0.3 did not equal the joint likelihood
  conditioned on the chosen observed time.

## Applied fixes

The session now registers floating dynamics for both amplitude groups, with
public names shared and Abar cache owners remapped. Model-wide and individual
component normalization settings are preserved. Projection caches evaluate
fitted dynamics, coefficients and scales, with efficiency/veto evaluated at
the rendering points.

Dalitz marginals and snapshots reuse the likelihood's true-time acceptance,
Gaussian convolution and selected-window integrals. Snapshots mix the jointly
normalized flavour PDFs before conditioning on time. Low-level diagnostics
require acceptance at the requested points for acceptance-weighted caches;
event-wise sigma_t or mistag arrays are rejected rather than implicitly
averaged. A specified distribution over those conditioning variables would
be needed to build a single marginal in that case.

The plotting tests now expect the existing English `best fit` label. The
model-update test constrains its otherwise unidentifiable global NR scale;
it no longer relies on numerical noise to produce a covariance.

## Validation

Regression tests compare session amplitudes against independent model caches,
check mass gradients against finite differences in both A and explicit Abar,
check normalization overrides and interference matrix blocks, and compare
rendered bins to a directly weighted fitted amplitude. Numerical integrations
of the joint PDF verify both tags' Dalitz marginals and conditional snapshots
with nonuniform efficiency, temporal acceptance, mistag and scalar resolution.
A coarse quadrature need not exactly cancel odd interference, so reflection
is checked at the amplitude/matrix level rather than assuming exactly equal
total intensities on an arbitrary finite grid.

Commands:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg pytest tests/test_time_dependent.py tests/test_time_dependent_workflow.py tests/test_plotting.py -q
JAX_PLATFORMS=cpu python benchmarks/benchmark_time_dependent.py --resolution 20 --events 10000 --repeats 5
```

All 81 selected tests passed; Ruff passed on the changed Python files.

The reduced three-resonance Asimov benchmark returned a valid fit with
x=0.005600009764 and y=0.002999952653 for generated x=0.0056 and y=0.003.
Value-plus-gradient evaluation took about 3.07 ms for 48,000 quadrature points
on the review machine (CPU); this is a timing observation, not a claim about
other hardware.
