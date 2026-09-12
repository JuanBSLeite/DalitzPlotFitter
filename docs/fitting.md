# Fitting and statistical validation

DalitzPlotFitter uses `iminuit` for minimization while JAX evaluates the objective and automatic gradient.

## NLL and Minuit convention

For a negative log-likelihood, Minuit uses

```text
errordef = 0.5
```

so HESSE one-parameter uncertainties correspond to `Delta NLL = 0.5`.

`Minimizer` uses a default tolerance of `1e-4`. Fit validity must not be judged from `valid` or EDM alone: always compare the fitted NLL with known reference points in closure tests and inspect pulls/covariance quality.

## Normalization samples

By default, amplitude-component and PDF normalization integrals use mass-plane
Gauss--Legendre or Square-Dalitz quadrature. An external `normalization_sample`
selects `toy-mc` and is reused unchanged during the fit. See
[Monte Carlo normalization](mc_integration.md) for proposal weights and selection.

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

where `generate_phase_space()` is used only to generate event/proposal samples. `prepare_cache()` uses the model-owned normalization sample unless an explicit integration sample is supplied.

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

To preserve the global convention while leaving one component unscaled, set
`normalize_component=False` on its `Resonance`, `NonResonant`, or
`DalitzAmplitude` declaration:

```python
qmi_component = Resonance(
    "S_wave_QMI",
    pair=(0, 2),
    coefficient=qmi_coefficient,
    lineshape=qmi,
    normalize_component=False,
)
```

The raw component is still included in the full Hermitian normalization matrix,
including its interference rows and columns, and therefore in the total PDF
normalization. `normalize_component=None` (the default) inherits the model-level
`normalize_components` setting; an explicit boolean overrides it.

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

### Fit fraction errors (delta method)

`fit_fractions()`/`print_fit_fractions()` only report central values --
`FF_i = Re(conj(c_i) M_ii c_i) / (c^dagger M c)`, a nonlinear function of the
coefficient vector `c` and the Hermitian normalization matrix `M_ij =
integral conj(F_i) F_j dPhi` (see "Normalization: the central invariant" in
`CLAUDE.md`). `M` itself is a function of every floating `DYNAMICS`
parameter, not just the coefficients: a component's mass, width, or (for
`QMI`) every knot magnitude/phase all change `M_ii`/`M_ij` and therefore
every fit fraction, including other components' fractions through the shared
denominator `c^dagger M c`. `fit_fraction_errors` reports the standard error
on this nonlinear function via the delta method, the standard first-order
error-propagation technique also implicit in Minuit's own HESSE parameter
errors: expand `f(theta)` (here the vector of fit fractions) to first order
around the postfit point `theta_hat`,

```text
f(theta) ~= f(theta_hat) + J (theta - theta_hat),   J = df/dtheta |_theta_hat
Cov(f)   ~= J Cov(theta_hat) J^T
```

and take `sigma(f_i) = sqrt(Cov(f)_ii)`. `Cov(theta_hat)` is Minuit's postfit
covariance (`result.covariance` after HESSE, from `Minimizer.fit`) -- the
same curvature-based, locally-Gaussian estimate already used for every plain
parameter's `sigma` printed elsewhere in this document. Propagating it
through fit fractions this way is therefore no less (and no more) justified
than trusting those parameter errors themselves; see "Caveats" below for when
that stops being a good approximation.

**How the Jacobian is computed.** Earlier revisions of this kind of
propagation (and the still-common approach elsewhere in the field) estimate
`J` by finite differences: nudge each parameter by a small step, recompute
`f`, and take a symmetric difference quotient -- exactly what
`docs/cp_coefficients.md`'s hand-rolled CP-observable Jacobian still does,
since that one differentiates a plain NumPy closed-form expression outside
JAX. `fit_fraction_errors` instead differentiates
`PreparedAmplitudeCache.fit_fractions` -- a pure JAX function of the full
parameter mapping, exactly like the objective itself -- with reverse-mode
autodiff (`dalitzplotfitter.observables.delta_method_jacobian`), giving `J`
to floating-point precision with no step-size tuning and no truncation
error. Reverse mode is also the efficient *direction* here: a fit fraction
vector has far fewer entries (one per component) than a QMI-heavy model has
free parameters -- `13_b2kkk_cpvfit_qmi.ipynb`'s S_QMI alone contributes on
the order of a hundred floating knot magnitudes/phases -- so the cost scales
with the output count, not with how many parameters float.

That said, `delta_method_jacobian` does *not* use `jax.jacrev`'s default
`vmap`-batched sweep over every output row at once: it takes one
`jax.vjp` linearization pass, then loops one cotangent-basis row through the
resulting backward function per output, in an ordinary Python loop. For a
handful of outputs this is a minor difference in *speed* -- but it is not a
minor difference in *memory*. `fit_fractions`'s reverse pass touches the
normalization-matrix computation for every floating dynamics parameter (a
QMI knot, in `13_b2kkk_cpvfit_qmi.ipynb`), which is comparatively heavy
because it is built from the full normalization sample; `vmap`-batching
that backward pass across every output row at once multiplies an already
sizeable per-row intermediate by the output count *simultaneously*, which
was enough to exhaust GPU memory outright for that notebook's joint B+/B-
Jacobian (16 outputs) before the loop replaced it. The loop keeps peak
memory bounded to one row's cost regardless of output count, trading it for
`len(output)` sequential backward passes instead of one vectorized one --
worthwhile whenever outputs are few and each backward pass is heavy, as
here.

```python
errors = model.fit_fraction_errors(fit_values, result.covariance)
```

Internally this resolves `fit_values` against every `Parameter` in
`model.parameters` (so a partial dict falls back to each parameter's own
declared value, exactly like `fit_fractions()`), builds the same physical- or
acceptance-weighted `PreparedAmplitudeCache` `print_fit_fractions()` would
(`efficiency=...` selects the convention), differentiates its
`fit_fractions(values)` with respect to every non-fixed parameter by default,
and propagates `result.covariance` through that Jacobian. `result.covariance`
can be passed directly: `delta_method_covariance` looks it up by parameter
name (`covariance[a, b]`) rather than assuming a particular positional order,
so which parameters happen to be free/fixed, or in what order Minuit stores
them, does not matter -- see "Pitfall" below for why that name lookup is
mandatory, not just convenient. `FitSession` exposes the same call as
`session.fit_fraction_errors(result)`.

**The CP joint case.** `CPFitSession.fit_fraction_errors(result)` does not
call the single-model version twice. `plus_model`/`minus_model` typically
share almost every fit parameter (every `CPRealImag` coefficient, every
dynamics parameter -- a QMI knot or a `GaussianConstraint`-anchored mass is
literally the same `Parameter` object for both charges), so `FF_plus` and
`FF_minus` are correlated, and that correlation matters for anything derived
from both of them together. The joint version instead stacks both charges'
fraction vectors into one function and differentiates it once:

```python
def joint_fractions(values):
    return jnp.concatenate([plus_cache.fit_fractions(values), minus_cache.fit_fractions(values)])

J = delta_method_jacobian(joint_fractions, values, parameter_names)   # shape (2n, m)
full_covariance = J @ covariance_matrix @ J.T                         # shape (2n, 2n)
```

one `delta_method_jacobian` call over the *union* of both models' free
parameter names, giving a single `(2n, m)` Jacobian whose top half is sensitivities of the `n`
B+ fractions and bottom half of the `n` B- fractions to the *same* `m`
parameters. `J @ C @ J.T` then gives the full `(2n, 2n)` covariance in one
step: its top-left and bottom-right `n x n` blocks are `Var(FF_plus)` and
`Var(FF_minus)` (what `errors["plus"]`/`errors["minus"]` report), and its
off-diagonal block's diagonal is `Cov(FF_plus_i, FF_minus_i)` for each
matching component `i`. That cross term is exactly what makes
`errors["mean"]` more than guesswork:

```text
FF_mean = 0.5*(FF_plus + FF_minus)
Var(FF_mean) = 0.25*(Var(FF_plus) + Var(FF_minus) + 2*Cov(FF_plus, FF_minus))
```

```python
errors = session.fit_fraction_errors(result)
errors["plus"]["rho0_770"]   # sigma(FF_plus) for one component
errors["minus"]["rho0_770"]  # sigma(FF_minus)
errors["mean"]["rho0_770"]   # sigma(FF_mean), including Cov(FF_plus, FF_minus)
```

Skipping the cross term (i.e. treating the two charges as independent and
adding their variances in quadrature) would silently overstate or understate
`sigma(FF_mean)` whenever `Cov(FF_plus, FF_minus)` is non-negligible -- which
it generally is here, precisely because the two charges share parameters
(see `docs/cp_coefficients.md`, "Derived-observable uncertainties").

**Pitfall: covariance ordering.** `delta_method_covariance`'s internal
`_covariance_matrix` helper always tries name-pair indexing
(`covariance[a, b]` for every `a, b` in the requested parameter names)
*before* ever treating `covariance` as a plain positionally-ordered array,
and only falls back to the latter if name indexing itself raises. This is
deliberate, not defensive boilerplate: an early version tried the array
interpretation first whenever `covariance`'s own shape matched
`(len(names), len(names))`, reasoning that a same-size object was "probably
already a dense matrix ordered like `names`". That reasoning breaks silently
for exactly the common case of requesting *every* free parameter, because
`iminuit`'s `Matrix` is itself array-convertible (`jnp.asarray(covariance)`
succeeds) *and* commonly has that exact same total dimension -- so the shape
check would "accidentally" pass and the code would use Minuit's own internal
parameter order instead of the caller's `parameter_names` order. The bug is
subtle to notice from symptoms alone: each fraction's *own* variance
(`v^T C v` for a single row `v` of `J`) is unaffected by a self-consistent
wrong permutation of `C`, so `errors["plus"]`/`errors["minus"]` still came
out correct -- only the *cross*-covariance between two different rows (e.g.
`Cov(FF_plus, FF_minus)`, or `Cov` between two different components) was
silently wrong. `tests/test_delta_method.py` has a regression test for this
exact scenario.

**Caveats.** This is a linear (Gaussian) approximation, not an exact
propagation:

- It is only as good as `Cov(theta_hat)` itself -- if HESSE did not run, did
  not converge, or `result.valid` is `False`, treat the propagated errors
  with the same skepticism as the raw parameter errors they are built from.
- It can understate the true uncertainty where `f(theta)` is strongly
  nonlinear near `theta_hat`, or where a fraction sits close to its hard
  lower bound of zero (a Gaussian is symmetric; a fraction is not, once its
  uncertainty band would otherwise cross zero). MINOS-style asymmetric scans
  or toy/bootstrap refits are the more robust (and far more expensive)
  alternative in that regime.
- Gaussian-constrained parameters (`GaussianConstraint`, e.g.
  `rho0_1450`/`rho0_1700`/`phi1020`/`chic0` mass and width in
  `13_b2kkk_cpvfit_qmi.ipynb`) need no special handling: the constraint is
  part of the NLL that HESSE differentiates, so its effect on
  `Cov(theta_hat)` -- and hence on every propagated fit-fraction error -- is
  already included automatically, whether or not that parameter is fixed.

`dalitzplotfitter.observables.delta_method_errors`/`delta_method_covariance`
are general-purpose: they work for any JAX-differentiable postfit quantity,
not just fit fractions (interference fractions, the CP-observable table in
`docs/cp_coefficients.md`, or any custom derived observable) -- see
`docs/catalog.md` ("Delta-method error propagation").
