# Time-dependent Dalitz implementation and Belle audit — 2026-09-23

## Implemented and tested

`NeutralMesonMixing` implements the printed rates in Eqs. (1–2) of
[Belle 2014](https://arxiv.org/abs/1404.2412). `TimeDependentDalitzNLL` uses one
prepared basis for the two amplitudes and their full complex overlap. It is
conditional on flavour tag, supports posterior mistag probabilities, exact
ideal-time normalization, and factorized true-time acceptance with numerical
Gaussian response. No old CPJointNLL or charged-meson notebook is changed.

CPU validation:

```
pytest tests/test_time_dependent.py tests/test_amplitude_cache.py \
       tests/test_dynamic_normalization_chunks.py tests/test_cp_likelihood.py -q
69 passed in 361.30s
```

Independent numerical tests cover Belle rate equations, SciPy integration and
Gaussian convolution, two-tag normalization, overlap conjugation, efficiency,
finite windows, mistags, all five mixing derivatives and floating dynamics.
Fixed lineshapes are forbidden from reevaluating in a dedicated cache test.
New module/tests/benchmark plus changed cache and likelihood exports pass Ruff.
The root package export file retains pre-existing lint debt.

`benchmark_time_dependent.py --resolution 40 --repeats 5` (CPU, 192,000 Asimov
points, 100,000 equivalent events, three-resonance demonstration):

- generated x=0.0056, y=0.0030 using independent NumPy rates;
- fitted x=0.00560002449, y=0.00300020915, valid minimum;
- preparation 0.986 s, value/gradient compile 0.959 s, evaluation 0.0259 s.

The optional discrete-quadrature toy, seed 20260923, 100,000 events, converged to
x=0.0029184 +/- 0.0032491 and y=0.0018689 +/- 0.0031155. One toy is a smoke test,
not a bias/pull-coverage study. Timing is machine-specific, not a speed guarantee.

## Full-component notebook

[Notebook](../../notebooks/benchmark/belle_2014_d0_kspipi_time_dependent.ipynb)
contains all 15 components, Belle's tabulated coefficients and S-wave parameters,
K-matrix production terms, and the corrected K-pi production formula from the
[BABAR 2010 supplement](https://arxiv.org/html/1004.5053). The existing LASS model
is not interchangeable with this formula. The local helper was checked at
0.70, 0.85, 1.10, 1.46 and 1.65 GeV against an independent complex-cotangent
expression (relative agreement 1e-12).

Notebook execution uses CPU and float64, DP resolutions 80/160 and 48 time nodes.
All code cells execute, both temporal rates agree with their NumPy reference,
and an Asimov x/y fit closes to 3e-6 absolute. Those checks validate the local
model and time implementation; they do not establish Belle-model equivalence.

## Remaining discrepancy

At DP resolution 160 the notebook gives FFs approximately:

| Component | Local [%] | Belle [%] |
|---|---:|---:|
| K*(892)- | 37.118 | 60.45 |
| rho(770) | 12.395 | 20.00 |
| pi-pi S | 30.487 | 12.88 |
| K2*(1430)- | 0.228 | 2.21 |

Changing resolution 80 to 160 shifts the total integral by -0.18% and the
largest FF by 0.30 percentage point. This is not a full convergence study, but
the large FF discrepancy remains. The notebook explicitly labels its working
choices for non-tabulated masses/widths, inherited radii, Zemach angular factors,
and K-matrix details; their exact mapping to Belle still needs verification.
Published coefficients have not been adjusted to force agreement. The notebook
is a runnable reproduction audit, **not a certified reproduction** of Belle's
full amplitude model or measured x/y uncertainties.

Experimental reproduction additionally requires event-level data, efficiency,
resolution mixtures, backgrounds and their fitted parameters. The current
signal-only API does not pretend to supply those inputs.
