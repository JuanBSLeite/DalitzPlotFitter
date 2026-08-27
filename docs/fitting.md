# Fitting and statistical validation

DalitzPlotFitter uses `iminuit` for minimization while JAX evaluates the objective and gradient.

## NLL convention

For a negative log-likelihood, Minuit uses

```text
errordef = 0.5
```

so HESSE one-parameter uncertainties correspond to `Delta NLL = 0.5`. `Minimizer` uses this convention by default. `Parameter.step` and parameter bounds are forwarded to Minuit.

## RealImag coefficients

The supported complex coefficient is

```text
c = x + i y
```

through `RealImag`. `x` and `y` may be constants or fit `Parameter` objects. There is no parallel magnitude/phase, Cartesian, or CP coefficient API in the current code base.

For the `D+ -> pi- pi+ pi+` reference model, the rho coefficient is fixed to

```text
c_rho = 1 + 0 i
```

and the remaining real and imaginary coordinates float.

## Closure criterion

For every floating coordinate, generated and fitted values are compatible when

```text
pull = (value_gen - value_fit) / sigma_fit
abs(pull) < 1
```

where `sigma_fit` is the HESSE uncertainty from the `errordef=0.5` NLL fit.

The reference end-to-end validation target is

```text
unweighted fit pseudo-data:     100,000 events
weighted normalization MC:    1,000,000 events
```

Pseudo-data are produced by weighted resampling from a larger `phasespace` candidate pool using

```text
w_target = w_PS |A(theta_gen)|^2.
```

The normalization sample is fixed during minimization so the objective remains deterministic.

## Cached normalization

For coefficient-only fits, the complete Laura++ component amplitudes are cached on data and normalization samples. The matrix

```text
M_ij = (1/N_MC) sum_k w_PS,k F_i*(x_k) F_j(x_k)
```

is cached and the normalization becomes

```text
N(c) = c^dagger M c.
```

When a dynamical parameter is later floated, only its owning component and the affected matrix row/column are reevaluated.
