# Fitting and statistical validation

DalitzPlotFitter uses `iminuit` as the minimizer while JAX evaluates the objective and its gradient.

## Negative-log-likelihood convention

For a negative log-likelihood (NLL), Minuit must use

```text
errordef = 0.5
```

so that HESSE parameter uncertainties correspond to a change

```text
Delta NLL = 0.5
```

for a one-parameter Gaussian approximation. `dalitzplotfitter.fit.Minimizer` therefore uses `errordef=0.5` by default. A different value can be supplied explicitly only when the minimized objective follows another statistical convention.

`Parameter.step` is forwarded to the corresponding Minuit initial step/error setting. Bounds are forwarded to `Minuit.limits`.

## Cartesian fit coefficients

For CP-conserving amplitude fits, `FitCartesian` parameterizes a complex coefficient directly as

```text
c = x + i y.
```

It contains only the two fit parameters `x` and `y` and has no flavor dependence or CP-violating offsets. This is distinct from `FitCartesianCP`, which contains the additional CP parameters `dx` and `dy`.

The reference `D+ -> pi- pi+ pi+` closure now uses Cartesian coefficients for the `f0` and non-resonant components. The rho reference coefficient is fixed to

```text
c_rho = 1 + 0 i.
```

The Cartesian representation avoids the periodic phase boundary of a magnitude/phase parameterization while describing exactly the same complex amplitudes.

## Toy-MC closure tests

A single finite toy sample is a statistical experiment. Closure should therefore not be judged only by a rigid absolute difference between fitted and generated values.

For each fitted parameter, define the pull

```text
pull = (theta_fit - theta_true) / sigma_fit
```

where `sigma_fit` is the HESSE uncertainty computed with the correct NLL `errordef=0.5` convention.

The reference closure test requires both:

1. a bounded pull, to test consistency with the fitted statistical uncertainty;
2. a broad absolute sanity bound, to prevent a wrong local minimum or pathological uncertainty estimate from passing merely because the reported uncertainty is very large.

This is intentionally different from loosening a failing absolute tolerance. The pull test uses the statistical scale predicted by the fit itself, while the absolute bound remains as an independent guardrail.

The current high-statistics reference closure uses:

```text
fit sample:           100,000 events
normalization sample: 1,000,000 events
```

The minimization is evaluated from several separated starts and the valid minimum with the lowest NLL is selected. The injected truth NLL is also compared with the best fitted NLL as an additional diagnostic.

## Monte Carlo normalization

Toy generation and fit normalization use independent phase-space samples. The fit normalization sample remains fixed during minimization so that the objective is deterministic.

For coefficient-only fits, component amplitudes and the normalization matrix are cached. If a dynamical parameter floats, only its owning component and the corresponding normalization-matrix rows/columns are recomputed.
