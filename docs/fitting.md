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

## RealImag fit coefficients

For CP-conserving amplitude fits, `RealImag` parameterizes a complex coefficient directly as

```text
c = x + i y.
```

The same `RealImag` class accepts either plain numerical values or fit `Parameter` objects. When `Parameter` objects are supplied, their current values are resolved from the minimizer mapping. There is no separate non-CP `FitCartesian` type.

This is distinct from CP-dependent coefficient parameterizations such as `CartesianCP`, which contain additional CP parameters.

The reference `D+ -> pi- pi+ pi+` closure uses `RealImag` coefficients for the rho, f0 and non-resonant terms, with the reference fixed to

```text
c_rho = 1 + 0 i.
```

## Toy-MC closure tests

The reference closure test checks statistical compatibility directly. For every fitted real or imaginary coefficient component,

```text
pull = (generated - fitted) / fitted_error
```

and the required condition is

```text
abs(pull) < 1.
```

The fitted error is the HESSE uncertainty obtained with the NLL convention `errordef=0.5`.

The current high-statistics reference closure uses:

```text
fit sample:           100,000 events
normalization sample: 1,000,000 events
```

The minimization is evaluated from several separated starts and the valid minimum with the lowest NLL is selected. The injected truth NLL is also compared with the best fitted NLL as an additional diagnostic.

## Monte Carlo normalization

Toy generation and fit normalization use independent phase-space samples. The fit normalization sample remains fixed during minimization so that the objective is deterministic.

For coefficient-only fits, component amplitudes and the normalization matrix are cached. If a dynamical parameter floats, only its owning component and the corresponding normalization-matrix rows/columns are recomputed.
