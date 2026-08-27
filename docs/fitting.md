# Fitting and statistical validation

DalitzPlotFitter uses `iminuit` for minimization while JAX evaluates the objective and gradient.

## NLL convention

For a negative log-likelihood, Minuit uses

```text
errordef = 0.5
```

so HESSE one-parameter uncertainties correspond to `Delta NLL = 0.5`. `Minimizer` uses this convention by default.

The Minuit EDM tolerance is also explicit. The fitter defaults to

```text
tolerance = 1e-6
```

rather than relying on Minuit's looser default. `Parameter.step` and parameter bounds are forwarded to Minuit.

## RealImag coefficients

The supported complex coefficient is

```text
c = x + i y
```

through `RealImag`. `x` and `y` may be constants or fit `Parameter` objects.

Example:

```python
x = Parameter.coefficient("rho.x", 0.8, owner="rho")
y = Parameter.coefficient("rho.y", 0.2, owner="rho")
coefficient = RealImag(x, y)
```

One complex coefficient should be fixed to remove the arbitrary global amplitude scale and phase.

## Floating dynamical parameters

Resonance mass, width and Blatt-Weisskopf radii may also be `Parameter` objects. Dynamical parameters have an `owner` equal to the resonance component name:

```python
mass = Parameter.dynamics(
    "rho.mass",
    0.760,
    owner="rho",
    bounds=(0.73, 0.81),
)
width = Parameter.dynamics(
    "rho.width",
    0.180,
    owner="rho",
    bounds=(0.10, 0.22),
)
```

`DecayModel.parameters` collects coefficient and dynamics parameters automatically. `DecayModel.prepare_cache()` uses the same list to configure optimized likelihood evaluation.

Spin remains a fixed discrete model choice.

## Cached normalization

For coefficient-only fits, component amplitudes are cached on data and normalization samples. The matrix

```text
M_ij = (1/N_MC) sum_k w_PS,k F_i*(x_k) F_j(x_k)
```

is cached and the normalization becomes

```text
N(c) = c^dagger M c.
```

When a dynamical parameter floats, only its owning component is reevaluated on the data and normalization samples. The affected row and column of the normalization matrix are recomputed.

The cache is tested numerically against direct amplitude and normalization evaluation at multiple values of floating mass, width and complex coefficients.

## Unbinned NLL

For unweighted data events, up to parameter-independent constants,

```text
NLL(theta)
 = -sum_n log |A(x_n;theta)|^2
   + N_data log N(theta).
```

The normalization uses a fixed weighted phase-space Monte Carlo sample so the objective remains deterministic during minimization.

## Multistart minimization

Amplitude likelihoods may contain local minima. A robust fit should therefore not depend on one fortunate initial point.

```python
minimizer = Minimizer(nll, model.parameters, tolerance=1e-6)
scan = minimizer.fit_multistart(
    n_starts=20,
    seed=314159,
    include_default=False,
    simplex=True,
)
result = scan.best
```

Each trial begins from an independently randomized parameter point. The best solution is selected only from valid finite minima using the lowest NLL. The injected truth is never used to select or seed the fit. HESSE is run on the selected best minimum.

Useful closure diagnostics include

```text
trial validity
trial NLL
trial EDM
NLL(truth)
NLL(best fit)
NLL(best fit) - NLL(truth)
```

If the best valid minimum remains significantly above `NLL(truth)`, minimization has not found the known closure-region solution. If the best fit has an equal or lower NLL but noticeably different parameters, the issue may instead be statistical fluctuation, parameter correlation or a likelihood degeneracy.

## Closure criterion

For every floating coordinate, generated and fitted values are compatible when

```text
pull = (value_gen - value_fit) / sigma_fit
abs(pull) < 1
```

where `sigma_fit` is the HESSE uncertainty from the `errordef=0.5` NLL fit.

The reference validation scale is

```text
unweighted fit pseudo-data:     100,000 events
weighted normalization MC:    1,000,000 events
```

Pseudo-data are produced by weighted resampling from a larger `phasespace` candidate pool using

```text
w_target = w_PS |A(theta_gen)|^2.
```

## E791 notebooks

`notebooks/01_e791_dplus_fit2_generation.ipynb` and `notebooks/02_fit_dynamic_parameters.ipynb` use the same Fit-2-based amplitude definition for

```text
D+ -> pi- pi+ pi+
```

with current DalitzPlotFitter dynamics conventions.

For consistency with E791's treatment of the parent decay form factor, the examples use

```text
parent_radius = 0
```

which makes the parent Blatt-Weisskopf factor unity.

The project RBW convention is

```text
1 / (m0^2 - m^2 - i m0 Gamma)
```

which differs by a constant minus sign from the propagator sign written in E791. The notebooks keep `rho(770)=1+0i` as the reference and translate this constant sign by shifting only the non-resonant phase by 180 degrees. The current covariant angular convention remains a project convention, so the notebooks should be described as Fit-2-based examples rather than exact historical reproductions of the E791 fitter.

In notebook 2 the rho(770) coefficient is fixed to `1 + 0 i`; the `RealImag` coefficients of the other components float. Among dynamical parameters only

```text
rho1450.mass
rho1450.width
```

float. The fit uses 20 randomized multistart trials and reports NLL, EDM, the selected minimum, generated-versus-fitted pulls and projection comparisons.
