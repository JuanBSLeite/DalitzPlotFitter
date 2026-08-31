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

Each pseudoexperiment stores:

- fitted values and HESSE errors;
- randomized starting values;
- fit validity;
- NLL at the fitted minimum;
- NLL evaluated at the injected truth point;
- EDM;
- number of Minuit function calls.

## Accessing the distributions

```python
sigma_x = result.values("sigma.x")
sigma_x_errors = result.errors("sigma.x")
sigma_x_pulls = result.pulls("sigma.x")
nll_values = result.nll
```

By default, parameter arrays contain only valid fits. Pass `valid_only=False` to
`values()` or `errors()` when all pseudoexperiments are required.

## Statistical summaries

```python
result.print_summary()
rows = result.summary()
gaussian = result.gaussian_fit("sigma.x")
```

For every free parameter and for the NLL distribution, the summary contains the sample
mean and sample standard deviation and the mean and width obtained from an unbinned
Gaussian maximum-likelihood fit. The Gaussian-fit parameter uncertainties and fit
validity are stored as well.

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
