# Discriminating variables and external constraints

## Factorized discriminating-variable PDFs

Amplitude fits may include observables beyond the Dalitz coordinates, such as a reconstructed parent mass, BDT response, PID discriminator, or isolation variable.

The basic implementation uses a factorized component model

\[
P_k(\Phi,x_1,\ldots,x_n)
=
P_k^{DP}(\Phi)\prod_a P_{k,a}(x_a),
\]

where `k` labels signal or a background category.  Each one-dimensional factor is normalized on its declared fit range, so the Dalitz and discriminating-variable normalizations remain separable.

Available basic models are:

- `Gaussian1D(mean, sigma, low, high)`;
- `Exponential1D(slope, low, high)`;
- `Histogram1D(edges, values)`;
- `FactorizedDensity(base_density, observables, pdfs)`.

Example:

```python
signal_full = FactorizedDensity(
    base_density=lambda values: signal_dp(data, values),
    observables={"mass": mass, "bdt": bdt},
    pdfs={
        "mass": Gaussian1D(mass_mean, 0.014, 5.20, 5.35),
        "bdt": Histogram1D(edges, signal_bdt_shape),
    },
)
```

The factorization is an analysis assumption.  Correlations between the Dalitz position and a discriminating variable, or between two discriminating variables, require a multidimensional model and are intentionally outside the basic implementation.

`FactorizedDensity` can be used independently for the signal and for every named background category, so combinatorial, partially reconstructed, and misidentified backgrounds can have distinct mass/BDT shapes.

## Gaussian external constraints

`GaussianConstraint` adds

\[
\Delta \mathrm{NLL}
=
\frac12\left(\frac{x-\mu}{\sigma}\right)^2
\]

up to an additive parameter-independent constant.

```python
constraint = GaussianConstraint(signal_fraction, mean=0.70, sigma=0.04)
constrained_nll = ConstrainedNLL(base_nll, constraint)
```

Any number of independent constraints can be supplied to `ConstrainedNLL`.  Typical uses include external signal fractions, background yields, efficiency nuisance parameters, and calibration quantities.

Correlated multivariate Gaussian constraints are not part of the basic implementation yet.

## Tutorial notebooks

- `10_b2kpipi_discriminating_variables.ipynb` demonstrates a joint Dalitz + reconstructed-mass + BDT fit with mass and BDT projections.
- `11_b2kpipi_gaussian_constraints.ipynb` compares constrained and unconstrained fits and plots the corresponding NLL scans.
