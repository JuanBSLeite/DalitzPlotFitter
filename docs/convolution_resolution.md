# PDF convolution and detector resolution

DalitzPlotFitter provides a generic one-dimensional convolution layer for observables whose reconstructed value differs continuously from the underlying true value.

For a true PDF `f(t)` and a conditional resolution kernel `R(x | t)`, the observed density is

\[
g(x)=\frac{\int f(t)R(x\mid t)\,dt}{\int_{x_\min}^{x_\max}dx\int f(t)R(x\mid t)\,dt}.
\]

The finite observed-range normalization is explicit. This is important whenever detector resolution migrates probability outside the fitted window.

## Gaussian resolution kernel

```python
from dalitzplotfitter import GaussianResolution1D

resolution = GaussianResolution1D(
    sigma=0.010,
    bias=0.0,
)
```

The kernel is Gaussian in the reconstructed observable around `true + bias`. `sigma` and `bias` may also be fit `Parameter` objects. The probability retained inside a finite observed interval is calculated analytically with the Gaussian CDF.

## Convolving an existing PDF

```python
from dalitzplotfitter import ConvolvedPDF1D, Gaussian1D

true_mass = Gaussian1D(
    mean=5.279,
    sigma=0.015,
    low=5.20,
    high=5.36,
)

reconstructed_mass = ConvolvedPDF1D(
    true_mass,
    GaussianResolution1D(sigma=0.010),
    true_low=5.20,
    true_high=5.36,
    observed_low=5.20,
    observed_high=5.36,
    order=96,
)
```

The integral over the true variable uses Gauss--Legendre quadrature and remains fully JAX-compatible. `order` controls the quadrature accuracy.

The resulting object follows the same callable convention as the existing one-dimensional discriminant PDFs:

```python
values = reconstructed_mass(mass, parameters)
```

and can therefore be used directly inside `FactorizedDensity`:

```python
density = FactorizedDensity(
    base_density=dalitz_density,
    observables={"mass": mass},
    pdfs={"mass": reconstructed_mass},
)
```

## Parameter-dependent resolution

```python
from dalitzplotfitter import Parameter

sigma_res = Parameter(
    "mass_resolution.sigma",
    0.010,
    bounds=(0.001, 0.050),
)

resolution = GaussianResolution1D(sigma=sigma_res)
```

The convolution and its finite-window normalization are recomputed from the supplied parameter values, so the resolution width can participate in a fit.

## Convolution versus SCF migration

`ConvolvedPDF1D` and `SquareDalitzSCFMap` solve different detector-resolution problems.

- `ConvolvedPDF1D` describes continuous smearing of a one-dimensional observable such as mass, decay time or another discriminating variable.
- `SquareDalitzSCFMap` describes migration between true and reconstructed regions of the two-dimensional Square Dalitz plane for self-cross-feed/misreconstruction.

A multidimensional Dalitz-resolution treatment should therefore be implemented as a migration kernel/operator rather than by naively applying two independent one-dimensional convolutions to Dalitz invariants.

## Current numerical implementation

For event values `x_i`, `ConvolvedPDF1D` evaluates the kernel matrix between the event values and the fixed Gauss--Legendre true nodes. This is efficient for ordinary discriminating-variable samples but scales as `N_events * order` in memory and arithmetic. Chunked evaluation and analytic convolution special cases can be added later if very large event samples require them.
