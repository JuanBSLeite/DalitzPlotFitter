# GenFit pseudoexperiment studies

`GenFit` is the built-in generate-and-fit driver for repeated Dalitz-plot closure studies.
It is intended for the standard workflow in which many independent pseudo-data samples
are generated from the same injected model and fitted with randomized starting values.

## Basic use

```python
from dalitzplotfitter import GenFit

study = GenFit(
    model,
    n_fits=500,
    sample_size=50_000,
    truth_values=truth,
    grid_resolution=1000,
    pool_size=1_000_000,
    start_range=(-2.5, 2.5),
    seed=791,
    pool_seed=2000,
)
result = study.run()
```

The current implementation is optimized for coefficient-only fits, matching the
standard E791 Fit-2 closure test. Floating dynamics parameters are rejected explicitly
for now. This allows the expensive amplitude evaluation of the candidate pool and the
normalization matrix to be cached once and reused across all pseudoexperiments.

Each pseudoexperiment stores fitted values and HESSE errors, randomized starting
values, Minuit convergence, GenFit acceptance, explicit rejection reasons, the fitted
NLL, the NLL at the injected truth point, EDM, and the number of Minuit function calls.

## Converged versus accepted fits

GenFit separates numerical minimizer convergence from the stricter quality selection
used in statistical closure distributions.

A fit is `converged` when Minuit reports a valid minimum and the final NLL is finite.
A converged fit is `accepted` only when all configured quality requirements also pass:

- EDM is finite and `EDM <= max_edm` (default `1e-3`);
- every fitted-parameter uncertainty is finite and strictly positive;
- a covariance matrix exists, is positive definite, and HESSE did not fail
  (`require_posdef_covar=True` by default);
- no fitted parameter is reported at a limit (`reject_at_limit=True` by default).

The criteria can be configured explicitly:

```python
study = GenFit(
    model,
    n_fits=500,
    sample_size=50_000,
    max_edm=1e-3,
    require_posdef_covar=True,
    reject_at_limit=True,
)
```

The result exposes both selections:

```python
result.n_converged
result.n_accepted
result.convergence_rate
result.acceptance_rate
result.converged_mask
result.accepted_mask
result.rejection_summary()
```

`valid`, `valid_mask`, `n_valid`, and `success_rate` are retained as backward-compatible
aliases for the accepted-fit selection.

## Accessing the distributions

```python
sigma_x = result.values("sigma.x")
sigma_x_errors = result.errors("sigma.x")
sigma_x_pulls = result.pulls("sigma.x")
nll_values = result.nll
```

By default, parameter arrays contain only accepted fits. Pass `valid_only=False` to
`values()` or `errors()` when all pseudoexperiments are required.

## Statistical summaries

```python
result.print_summary()
rows = result.summary()
gaussian = result.gaussian_fit("sigma.x")
```

`print_summary()` reports the number of converged and accepted fits separately and
lists the accumulated rejection reasons before the parameter table.

For every free parameter and for the NLL distribution, the summary contains the sample
mean and sample standard deviation and the mean and width obtained from an unbinned
Gaussian maximum-likelihood fit. Only accepted fits enter these distributions.

## Robust outlier treatment

Numerical-quality rejection and statistical outlier rejection are separate steps.
After the accepted-fit selection, the robust helpers can remove catastrophic tails from
a one-dimensional parameter or NLL distribution using a median/MAD criterion:

```python
from dalitzplotfitter import print_genfit_robust_summary

print_genfit_robust_summary(result, threshold=5.0)
```

The printed table includes the number and percentage of rejected statistical outliers.
Raw pseudoexperiment results remain stored in `result`.

## Plots

```python
result.plot("sigma.x", bins=30)
result.plot("nll", bins=30)
figures = result.plot_all(bins=30)
```

The parameter histograms include the injected truth value as a vertical line and the
Gaussian fit as an overlay. `plot_all()` returns a dictionary of Matplotlib figures so
the caller can further customize or save them.

## Reproducibility

`seed` controls the independent pseudo-data samples and randomized starting points.
`pool_seed` controls the common phase-space candidate pool. Keeping both fixed makes a
GenFit study reproducible.
