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

## Toy-MC closure tests

A single finite toy sample is a statistical experiment. Closure should therefore not be judged only by a rigid absolute difference between fitted and generated values.

For each fitted parameter, define the pull

```text
pull = (theta_fit - theta_true) / sigma_fit
```

where `sigma_fit` is the HESSE uncertainty computed with the correct NLL `errordef=0.5` convention.

For angular parameters, the numerator is evaluated with the phase difference wrapped into the principal interval.

The reference closure test requires both:

1. a bounded pull, to test consistency with the fitted statistical uncertainty;
2. a broad absolute sanity bound, to prevent a wrong local minimum or pathological uncertainty estimate from passing merely because the reported uncertainty is very large.

This is intentionally different from loosening a failing absolute tolerance. The pull test uses the statistical scale predicted by the fit itself, while the absolute bound remains as an independent guardrail.

## Monte Carlo normalization

Toy generation and fit normalization use independent phase-space samples. The fit normalization sample remains fixed during minimization so that the objective is deterministic.

For coefficient-only fits, component amplitudes and the normalization matrix are cached. If a dynamical parameter floats, only its owning component and the corresponding normalization-matrix rows/columns are recomputed.
