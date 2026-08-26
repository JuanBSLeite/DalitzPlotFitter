# Symbolic line shapes and component normalization

DalitzPlotFitter is moving toward project-owned resonance dynamics based on the conventions used by Laura++ while retaining QRules/AmpForm for decay topology, angular structure and symmetrization.

## Architecture

The intended model flow is

```text
QRules
  -> decay topology and quantum-number validation
  -> AmpForm helicity/angular structure and symmetrization
  -> DalitzPlotFitter symbolic resonance dynamics
  -> SymPy expression
  -> TensorWaves
  -> JAX
  -> cached likelihood and gradients
```

The line-shape expression is constructed symbolically once. Fit parameters such as resonance mass, width, coupling constants and meson radius remain numerical arguments of the compiled JAX function. The symbolic expression is therefore **not rebuilt during minimization**.

## Laura++ relativistic Breit-Wigner

The first native dynamics implementation is the Laura++-convention relativistic Breit-Wigner

```text
R(m) = 1 / (m0^2 - m^2 - i m0 Gamma(m))
```

with

```text
Gamma(m) = Gamma0 (q/q0)^(2L+1) (m0/m) X(q r)^2.
```

The two-body breakup momentum is

```text
q(m) = sqrt(lambda(m^2, m1^2, m2^2)) / (2 m),
```

where

```text
lambda(x,y,z) = x^2 + y^2 + z^2 - 2xy - 2xz - 2yz.
```

The Blatt-Weisskopf factor is normalized at the resonance pole so that

```text
X(q0 r) = 1.
```

The implementation currently supports orbital angular momenta `L=0..5` for the Laura++ Blatt-Weisskopf polynomials.

## Using project-owned dynamics with AmpForm

`AmplitudeBuilder` now accepts

```python
model = AmplitudeBuilder(
    reaction,
    resonance_dynamics="laura",
).build()
```

This replaces only the resonance dynamics builder. AmpForm still supplies the topology, helicity/angular factors and identical-particle symmetrization.

The previous behavior remains available with

```python
model = AmplitudeBuilder(
    reaction,
    resonance_dynamics="ampform",
).build()
```

This compatibility mode is retained while the new dynamics are validated against AmpForm and Laura++.

## Individual component normalization

Laura++ normalizes each dynamical amplitude term over the Dalitz plot. DalitzPlotFitter supports the same convention through

```python
cache = PreparedAmplitudeCache.prepare(
    components,
    data=data,
    normalization_data=normalization_data,
    normalization_weights=normalization_weights,
    normalize_components=True,
)
```

For each complete dynamical component `F_i(x)`, DalitzPlotFitter evaluates

```text
N_i = integral epsilon(x) |F_i(x)|^2 dPhi
```

on the reference normalization sample and uses

```text
F_i_normalized(x) = F_i(x) / sqrt(N_i).
```

Consequently the diagonal of the cached normalization matrix satisfies

```text
M_ii = 1
```

up to Monte Carlo precision.

### What is normalized

The normalization applies to the **complete component amplitude**, not merely to the one-dimensional resonance function `R(m)`. Thus it includes whatever is present in `F_i`, for example:

- resonance line shape;
- angular term;
- Blatt-Weisskopf factors;
- identical-particle symmetrization;
- any other dynamics included in that component.

This is the useful convention for making external complex coefficients numerically comparable.

Efficiency is included in the component normalization when `efficiency_normalization` is supplied to the cache, consistently with the signal PDF used by the fitter.

## Floating dynamical parameters

If only complex coefficients change, normalized component amplitudes and the full normalization matrix remain cached.

If a dynamical parameter of component `i` changes, for example

```text
rho.mass
rho.width
rho.radius
```

DalitzPlotFitter reevaluates only that component on data and normalization MC. If component normalization is enabled, its normalization factor `N_i` is also recomputed. Only the corresponding normalization-matrix row and column are updated.

Other static components remain cached.

## Planned native dynamics

The implementation sequence is:

1. Laura++ relativistic Breit-Wigner — implemented;
2. Gounaris-Sakurai — planned, especially for `rho(770)`;
3. Flatte — planned, especially for `f0(980)` near the `K Kbar` threshold;
4. LASS — planned for `K pi` S-wave models;
5. more complex models such as K-matrix after the simpler line shapes have closure and reference-validation tests.

The `D+ -> pi- pi+ pi+` reference model will be migrated in controlled steps. The first validation compares the native RBW infrastructure with the existing AmpForm benchmark. The physically improved reference model is intended to use Gounaris-Sakurai for the `rho(770)` and Flatte for the `f0(980)`.

## Validation requirements

Every native line shape should have unit tests for its analytic limits and dedicated numerical comparison tests against a trusted reference implementation or published convention. Examples include

```text
q(m0) = q0
Gamma(m0) = Gamma0
X(q0 r) = 1
```

and, when component normalization is enabled,

```text
M_ii = 1.
```

New line shapes should also receive toy-MC closure tests before being considered production-ready.

## Reference

The conventions are based on:

J. Back et al., **Laura++: a Dalitz plot fitter**, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.
