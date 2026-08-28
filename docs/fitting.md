# Fitting and statistical validation

DalitzPlotFitter uses `iminuit` for minimization while JAX evaluates the objective and gradient.

## NLL and Minuit convention

For a negative log-likelihood, Minuit uses

```text
errordef = 0.5
```

so HESSE one-parameter uncertainties correspond to `Delta NLL = 0.5`.

The EDM tolerance is explicit in `Minimizer`. Multistart trial minima use Minuit strategy 1. The selected minimum is rerun with the careful strategy 2 before HESSE because amplitude fits can contain strong correlations between complex coefficients and dynamical shape parameters.

## Component normalization is the default amplitude convention

Every dynamical component is normalized by default according to

```text
integral dPhi |F_j|^2 = 1.
```

The normalization is applied before multiplication by the complex coefficient:

```text
A = sum_j c_j F_j_normalized.
```

This is especially important for floating lineshape parameters. Without per-component normalization, changing a width also changes the overall scale of the raw Breit-Wigner and produces an avoidable correlation between `|c_j|` and the width.

`DecayModel` owns a weighted phase-space normalization sample with public defaults

```python
model = DecayModel(
    channel,
    components,
    normalize_components=True,
    normalization_size=1_000_000,
    normalization_seed=2027,
)
```

The sample is generated lazily on first use and then reused. The common fit workflow is

```python
data = ...
cache = model.prepare_cache(data)
pdf = model.pdf()
```

without manually generating a second normalization sample.

Raw components can be requested explicitly with

```python
DecayModel(..., normalize_components=False)
```

and the MC precision can be changed with `normalization_size`.

Detector efficiency is deliberately excluded from the individual component normalization. It enters only the total signal-PDF normalization.

For a floating dynamical parameter the component scale is recalculated at each parameter point:

```text
F_j(x;theta)
 -> F_j(x;theta) / sqrt(<w_PS |F_j(theta)|^2>).
```

## Fit-performance path

Floating lineshape parameters used to be expensive because every likelihood evaluation rebuilt the full covariant kinematics for the affected component on both the data and normalization samples.

The optimized cache now separates event kinematics from dynamical parameters. For each resonance pairing it prepares once

```text
m_ij
p*
p
q
cos(theta)
```

from the Dalitz invariants. These quantities do not depend on the resonance pole mass or width. During a fit of `m0` or `Gamma0`, repeated likelihood evaluations therefore recompute only the genuinely parameter-dependent pieces:

```text
q0, p0
Blatt-Weisskopf pole factors
running width
lineshape
component normalization
interference row/column
```

The invariant reconstruction is regression-tested against the original Lorentz-boost calculation.

For a dynamic component, the affected normalization-matrix row is also evaluated with one vectorized reduction instead of one full pass over the normalization sample per matrix element.

JAX still incurs a one-time compilation cost on the first objective/gradient evaluation. Timing should therefore distinguish

```text
cache preparation / first JIT compilation
steady-state likelihood evaluations
full Minuit runtime
```

The default multistart recommendation uses gradient-based MIGRAD directly:

```python
scan = minimizer.fit_multistart(
    n_starts=20,
    seed=314159,
    include_default=False,
    simplex=False,
)
```

`simplex=True` remains available as a fallback for difficult starting points, but running Simplex before every one of many starts is substantially more expensive and is not the normal path now that the analytic JAX gradients are validated.

## RealImag coefficients

The supported complex coefficient is

```text
c = x + i y
```

through `RealImag`. One complex coefficient should be fixed to remove the arbitrary global amplitude scale and phase.

## Floating dynamical parameters

Mass, width and Blatt-Weisskopf radii may be `Parameter` objects. Dynamics parameters must have an `owner` equal to their amplitude-component name.

```python
mass = Parameter.dynamics(
    "rho.mass", 0.760, owner="rho", bounds=(0.73, 0.81)
)
width = Parameter.dynamics(
    "rho.width", 0.180, owner="rho", bounds=(0.10, 0.22)
)
```

`DecayModel.parameters` collects coefficient and dynamics parameters automatically. When a dynamical parameter changes, `PreparedAmplitudeCache` reevaluates only its owning component on data and normalization samples and recomputes the affected rows and columns of the normalization matrix.

## Dynamic-path consistency tests

Floating shape parameters receive dedicated tests in `tests/test_dynamic_fit_consistency.py`.

The validation deliberately separates implementation correctness from statistical identifiability:

1. JAX mass/width gradients are compared with central finite differences.
2. A discrete Asimov construction uses the same phase-space support for the truth distribution and normalization; the gradient at the injected truth must vanish.
3. With an identifiable resonance and fixed coefficient, a multistart fit of mass and width must recover the injected values from displaced starts.
4. Cached intensity and normalization are independently checked against direct model evaluation at multiple dynamic-parameter points.

These tests are designed to expose errors in

```text
Parameter -> ResonanceContext -> lineshape -> component amplitude
          -> component normalization -> normalization cache
          -> JAX gradient -> Minuit
```

rather than relying only on one pseudoexperiment closure.

## Monte Carlo normalization and floating shapes

For unweighted data,

```text
NLL(theta)
 = -sum_n log |A(x_n;theta)|^2
   + N_data log N(theta)
```

with

```text
N(theta) ~= (1/N_MC) sum_k w_PS,k |A(x_k;theta)|^2.
```

A fixed normalization sample keeps the objective deterministic, but finite-MC integration error remains. This is more important when masses, widths or other lineshape parameters float because their likelihood gradients contain derivatives of the normalization integral itself.

For reference closure studies the default internal normalization sample contains

```text
normalization MC: 1,000,000 events
```

while the E791 examples use

```text
pseudo-data:             100,000 events
candidate generation:  1,000,000 events
```

The candidate-generation pool is independent of the model-owned normalization MC.

## Multistart minimization

```python
minimizer = Minimizer(nll, model.parameters)
scan = minimizer.fit_multistart(
    n_starts=20,
    seed=314159,
    include_default=False,
    simplex=False,
)
result = scan.best
```

The injected truth is never used to seed or select the fit. Useful diagnostics are

```text
trial validity
trial NLL
trial EDM
NLL(truth)
NLL(best)
NLL(best) - NLL(truth)
correlation matrix / profile scans
```

If `NLL(best)` remains significantly above `NLL(truth)`, the minimizer did not find the known closure-region solution. If the best fit has equal or lower NLL but very different parameters, investigate statistical fluctuations, correlations, weak identifiability and normalization-MC precision.

## Closure criterion

For each floating coordinate,

```text
pull = (value_gen - value_fit) / sigma_fit
abs(pull) < 1
```

is the reference one-pseudoexperiment compatibility check. A single pull outside one standard deviation is not by itself evidence of fitter bias; bias studies require ensembles of pseudoexperiments.

## E791 examples and conventions

The E791 notebooks use the Fit-2 resonance content for

```text
D+ -> pi- pi+ pi+
```

with current DalitzPlotFitter angular conventions.

Historical E791 three-pion analyses used effective Blatt-Weisskopf radii

```text
parent_radius = 3.0 GeV^-1
resonance_radius = 3.0 GeV^-1
```

for the parent and resonance factors.

The project RBW convention

```text
1 / (m0^2 - m^2 - i m0 Gamma)
```

is the negative of the propagator sign written by E791. With `rho(770)=1+0i` retained as the reference coefficient, the examples account for the relative sign to the constant non-resonant term by shifting the NR phase by 180 degrees.

All notebook plots that decompose the model into amplitude components use the normalized component columns from `PreparedAmplitudeCache`.

`notebooks/02_fit_dynamic_parameters.ipynb` intentionally floats the mass and width of `rho1450` as a difficult stress test. This component has only about 0.7% Fit-2 fraction, so its shape parameters are intrinsically weakly constrained.

`notebooks/03_lineshape_parameter_diagnostics.ipynb` compares the dominant sigma and weak rho(1450) cases and studies normalization-MC precision.
