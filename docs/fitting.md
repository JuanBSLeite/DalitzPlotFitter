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

rho = Resonance(
    "rho",
    pair=(0, 1),
    coefficient=RealImag(x, y),
    mass=mass,
    width=width,
    spin=1,
)
```

`DecayModel.parameters` collects coefficient and dynamics parameters automatically. `DecayModel.prepare_cache()` uses the same list to configure optimized likelihood evaluation.

Dynamics plugins are also parameter-aware. Dataclass fields containing `Parameter` objects are resolved for each likelihood evaluation. This allows future plugins such as `Flatte`, `LASS`, or other lineshapes to expose fit parameters without changing `DecayModel`.

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

When a dynamical parameter floats, only its owning component is reevaluated on the data and normalization samples. The affected row and column of the normalization matrix are then recomputed; unrelated fixed components remain cached.

## Unbinned NLL

For unweighted data events,

```text
-NLL = sum log P(x; theta)
```

or equivalently, up to parameter-independent constants,

```text
NLL(theta)
 = -sum_n log |A(x_n;theta)|^2
   + N_data log N(theta).
```

The normalization is evaluated with a fixed weighted phase-space Monte Carlo sample so the objective remains deterministic during minimization.

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

## Fit notebook

`notebooks/02_fit_dynamic_parameters.ipynb` duplicates the E791 Fit 2 amplitude model used in notebook 1 for

```text
D+ -> pi- pi+ pi+
```

The rho(770) coefficient is fixed to `1 + 0 i` as the reference amplitude. The `RealImag` coefficients of the other Fit 2 components float. Among dynamical parameters, only

```text
rho1450.mass
rho1450.width
```

float; all other resonance masses and widths remain fixed to the values used in the generation example.

Before minimization, every free parameter is randomized uniformly inside its configured bounds. A fixed random seed is used only to keep the notebook reproducible, and the randomized point is passed explicitly to `Minimizer.fit(start_values=...)`.

The notebook uses 100,000 pseudo-data events and 1,000,000 normalization events, prints generated-versus-fitted pulls, and compares the full unlike-sign pion projection before and after the fit, including a rho(1450)-sensitive zoom.
