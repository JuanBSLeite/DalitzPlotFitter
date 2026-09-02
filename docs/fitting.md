# Fitting and statistical validation

DalitzPlotFitter uses `iminuit` for minimization while JAX evaluates the objective and automatic gradient.

## NLL and Minuit convention

For a negative log-likelihood, Minuit uses

```text
errordef = 0.5
```

so HESSE one-parameter uncertainties correspond to `Delta NLL = 0.5`.

`Minimizer` uses a default tolerance of `1e-4`. Fit validity must not be judged from `valid` or EDM alone: always compare the fitted NLL with known reference points in closure tests and inspect pulls/covariance quality.

## Deterministic normalization only

All amplitude-component and PDF normalization integrals use either mass-plane
Gauss--Legendre quadrature or Square-Dalitz quadrature. There is no Monte Carlo,
equal-area, or adaptive normalization path in the supported API.

The default model configuration is

```python
model = DecayModel(
    channel,
    components,
    normalize_components=True,
    normalization_method="gauss-legendre",
    normalization_bin_width=0.005,
)
```

The quadrature is constructed lazily and reused for the lifetime of the model.
This mass-plane method is the same base Gauss--Legendre prescription used by
Laura++.

The normal fit workflow is simply

```python
data = model.generate_phase_space(...)
cache = model.prepare_cache(data)
```

where `generate_phase_space()` is used only to generate event/proposal samples. `prepare_cache()` uses the deterministic model-owned grid unless an explicit grid sample is supplied.

For the default method, convergence should be checked by changing
`normalization_order_m13` and `normalization_order_m23`. For
`normalization_method="square-dalitz"`, change `normalization_resolution`.

```text
400 -> 600 -> 800 -> 1000 -> 1200.
```

## Component normalization convention

Every dynamical component is normalized by default according to

```text
integral dPhi |F_j|^2 = 1.
```

The normalization is applied before multiplication by the complex coefficient:

```text
A = sum_j c_j F_j_normalized.
```

For a floating dynamical parameter the scale is recalculated at the current parameter point using the deterministic grid:

```text
F_j(x;theta)
 -> F_j(x;theta) / sqrt(integral_grid |F_j(theta)|^2).
```

Detector efficiency is deliberately excluded from individual component normalization and enters only the total signal-PDF normalization.

Raw components can be requested explicitly with

```python
DecayModel(..., normalize_components=False)
```

## Cached normalization matrix

For linear complex coefficients,

```text
A(x) = sum_i c_i F_i(x)
```

and

```text
N(c) = integral |A|^2 dPhi = c^dagger M c,
```

with

```text
M_ij = integral conj(F_i) F_j dPhi.
```

For coefficient-only fits the component values and the full Hermitian normalization matrix are cached once. Changing magnitudes/phases therefore does not reevaluate lineshapes or reintegrate the Dalitz plot.

When a resonance mass, width, or another dynamical parameter floats, only the affected component and the corresponding normalization-matrix row/column are reevaluated.

## Fit-performance path

Parameter-independent event kinematics are prepared once:

```text
m_ij
p*
p
q
cos(theta)
```

During repeated likelihood evaluations only genuinely parameter-dependent quantities are recomputed, such as

```text
q0, p0
Blatt-Weisskopf pole factors
running width
lineshape
component normalization
interference row/column.
```

JAX has a one-time compilation cost on the first objective/gradient evaluation. Timing studies should distinguish cache preparation, first JIT compilation, steady-state likelihood evaluation, and full Minuit runtime.

## Gradient validation

`Minimizer.check_gradient()` compares the same JAX gradient supplied to Minuit with central finite differences:

```python
gradient_check = minimizer.check_gradient(
    start_values,
    step_scale=1e-5,
    print_table=True,
)
```

This should be used when introducing a new dynamical parameter or lineshape.

## RealImag coefficients

The supported complex coefficient parameterization is

```text
c = x + i y
```

through `RealImag`. One complex coefficient is normally fixed to remove the arbitrary global amplitude scale and phase.

## Floating dynamical parameters

Mass, width and other dynamics quantities may be `Parameter` objects. Dynamics parameters must have an `owner` equal to their amplitude-component name.

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

Meson/Blatt-Weisskopf radii are fixed by default unless explicitly promoted to fit parameters.

## Multistart minimization

```python
minimizer = Minimizer(nll, model.parameters, verbose=1)
scan = minimizer.fit_multistart(
    n_starts=20,
    seed=314159,
    include_default=False,
    simplex=False,
)
result = scan.best
```

For the standard closure notebooks we intentionally use one randomized start unless the study is explicitly about multistart robustness.

Useful diagnostics are

```text
validity
EDM
NLL(truth)
NLL(start)
NLL(fit)
NLL(fit) - NLL(truth)
pulls
covariance/correlation matrix.
```

A result with tiny EDM but `NLL(fit)` far above `NLL(truth)` is a failed closure even if Minuit reports `valid=True`.

## Closure criterion

For each floating coordinate,

```text
pull = (value_fit - value_gen) / sigma_fit.
```

A single pseudoexperiment is a closure check, not a bias measurement. Bias requires an ensemble of toys.

## E791 examples and conventions

The E791 notebooks use the Fit-2 resonance content for

```text
D+ -> pi- pi+ pi+.
```

Historical E791 three-pion analyses used effective Blatt-Weisskopf radii

```text
parent_radius = 3.0 GeV^-1
resonance_radius = 3.0 GeV^-1.
```

The project RBW convention is

```text
1 / (m0^2 - m^2 - i m0 Gamma).
```

With `rho(770)=1+0i` retained as the reference coefficient, the E791 examples account for the propagator-sign convention by shifting the constant non-resonant phase by 180 degrees.

The canonical E791 workflows are
`notebooks/01_e791_toy_fit.ipynb` and
`notebooks/02_e791_efficiency_background_fit.ipynb`.

## Fit fractions

After a fit, convert the Minuit values to a mapping and print the fractions:

```python
fit_values = {name: float(result.values[name]) for name in result.parameters}
model.print_fit_fractions(fit_values, include_interference=True)
```

By default this reports physical fractions. Pass the same efficiency callable
used in the likelihood through `efficiency=...` to report acceptance-weighted
fractions. The returned dictionary stores fractions as numbers rather than
percentages; the printed table uses percentages.
