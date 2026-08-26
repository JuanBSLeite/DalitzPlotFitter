# DalitzPlotFitter

DalitzPlotFitter is a Python package under development for unbinned amplitude fits of three-body decays. It is built around **QRules**, **AmpForm**, **TensorWaves/JAX** and **mplhep**.

The project deliberately uses one numerical backend: **JAX**. JAX is an internal implementation choice and is not included in public class names.

## Design goals

- construct decay chains and validate quantum numbers with QRules;
- formulate symbolic amplitudes with AmpForm;
- compile numerical amplitudes to JAX through TensorWaves;
- keep complex fit coefficients owned by DalitzPlotFitter rather than by AmpForm;
- allow both coefficient parameters and dynamical line-shape parameters to float in fits;
- keep Blatt-Weisskopf meson radii fixed by default and float them only on explicit request;
- cache fixed component amplitudes on data and normalization samples;
- use cached normalization matrices for coefficient-only fits;
- perform deterministic Monte Carlo normalization on a fixed three-body phase-space sample generated natively with JAX;
- optionally include efficiency in the signal PDF;
- model background with analytic/JAX callables or Dalitz histograms;
- support signal-only weighted likelihoods with sWeights;
- support simultaneous particle/antiparticle fits with CP violation;
- calculate fit fractions, interference fractions and CP-asymmetry observables;
- use iminuit while JAX evaluates the NLL and gradients;
- validate fitter changes with toy-MC closure tests;
- provide HEP-style diagnostics and Dalitz plots with mplhep.

## Coherent amplitude convention

DalitzPlotFitter uses AmpForm to construct the dynamical functions but owns the complex coefficients itself:

```text
A(x) = sum_i c_i F_i(x)
```

For an AmpForm resonance component, `compile_amplitude_component()` removes the AmpForm-generated helicity coupling `C_...` by setting it to unity. The resulting function is therefore only `F_i(x)`. A DalitzPlotFitter `AmplitudeComponent` then applies a coefficient object such as `MagPhase`, `CartesianCP`, or another Laura++ parameterisation.

This separation is essential for CP-violating simultaneous fits because the same dynamical component can be multiplied by different particle and antiparticle coefficients without modifying AmpForm.

## Fit parameters and caching

Fit parameters are first-class objects. They can represent coefficient parameters such as magnitudes and phases, or dynamical parameters such as resonance masses and widths.

Blatt-Weisskopf meson radii are **fixed by default**. AmpForm introduces the symbolic radius `d_res` with default value `1` in its form-factor and energy-dependent-width builders. DalitzPlotFitter keeps that value fixed unless the user explicitly creates a floating meson-radius parameter. A fixed radius therefore never invalidates the line-shape cache during minimization. If explicitly floated, it is treated as a dynamical parameter owned by the corresponding amplitude and invalidates only that component.

For a coefficient-only fit, the component amplitudes

```text
F_i(x_n)
```

are evaluated once on the data and once on the normalization sample. Every likelihood call then requires only the coherent linear combination

```text
A_n = sum_i c_i F_i(x_n)
```

rather than reevaluating Breit-Wigner functions, angular terms or form factors.

The normalization matrix

```text
M_ij = integral epsilon(x) F_i*(x) F_j(x) dPhi
```

is also cached, so the normalization for new coefficients is only

```text
I(c) = c^dagger M c.
```

If a dynamical parameter of one component floats, only that component is reevaluated. The cache updates only the affected normalization-matrix row and column; static components and static matrix blocks are reused.

## CP-violating coefficients

The coefficient API follows Table 1 of the Laura++ paper [J. Back et al., arXiv:1711.09854](https://arxiv.org/abs/1711.09854). The implemented parameterisations are `MagPhase`, `RealImag`, `BelleCP`, `CartesianCP`, `CartesianGammaCP`, `CleoCP`, `MagPhaseCP`, `PolarGammaCP`, `RealImagCP` and `RealImagGammaCP`.

`Flavor.PARTICLE` uses the `+` convention and `Flavor.ANTIPARTICLE` uses the `-` convention of the Laura++ table.

```python
from dalitzplotfitter import CartesianCP, Flavor

coefficient = CartesianCP(x=1.0, y=0.2, dx=0.03, dy=-0.01)
c_particle = coefficient.value(Flavor.PARTICLE)
c_antiparticle = coefficient.value(Flavor.ANTIPARTICLE)
```

## Three-body phase space

The core package does not depend on the TensorFlow-based `phasespace` package. Instead, `ThreeBodyPhaseSpace` generates weighted points directly in Dalitz coordinates using JAX and reconstructs a deterministic mother-rest-frame four-momentum configuration in the `(E, px, py, pz)` convention expected by AmpForm/TensorWaves.

The raw reconstruction is followed by a fixed global spatial rotation. This leaves every invariant mass and relative decay angle unchanged, but avoids placing a two-body subsystem exactly on a coordinate axis. That prevents artificial helicity-coordinate singularities in AmpForm for perfectly valid Dalitz points. This convention is physically harmless for the current unpolarized scalar-mother use case.

```python
import jax
from dalitzplotfitter import ThreeBodyPhaseSpace

phase_space = ThreeBodyPhaseSpace(
    mother_mass=1.86966,
    masses=(0.13957, 0.13957, 0.13957),
)
sample = phase_space.generate(jax.random.key(7), size=1_000_000)
```

When a QRules reaction is available, masses can be taken directly from it:

```python
phase_space = ThreeBodyPhaseSpace.from_reaction(reaction)
```

`ThreeBodyPhaseSpace.from_unit_square()` exposes the same deterministic mapping used internally by the random generator. The map takes points in `[0,1]^2` to physical Dalitz coordinates and returns the exact Jacobian `ds12 ds23 = w_PS du1 du2`. This shared parametrization is used by the toy-generator envelope search.

The same phase-space sample should be reused throughout a fit, making Monte Carlo normalization deterministic.

## D+ -> pi- pi+ pi+ reference model

The reference benchmark uses the explicit final-state ordering

```text
1 = pi-
2 = pi+_1
3 = pi+_2
```

so that

```text
s12 = m2(pi- pi+_1)
s13 = m2(pi- pi+_2)
s23 = m2(pi+_1 pi+_2)
```

The two positive pions are identical. QRules keeps indistinguishable quantum-state transitions and AmpForm restores the kinematically distinct `pi- pi+` pairings inside the amplitude. `KinematicTransformer` registers the corresponding topology permutations so all required invariant masses and helicity angles are generated automatically.

The minimal rho-only benchmark is:

```bash
python examples/01_dplus_rho.py
```

The first coherent multi-component benchmark is:

```bash
python examples/02_dplus_rho_f0_nr.py
```

and uses

```text
A = c_rho F_rho(770) + c_f0 F_f0(980) + c_NR
```

with `rho(770)0` fixed as the global magnitude/phase reference. The `f(0)(980)` component currently uses the default relativistic Breit-Wigner infrastructure as an integration benchmark. A Flatte parameterisation should be used as the dedicated physics option near the KK threshold and is a planned dynamics extension.

The multi-component example explicitly compares coherent and incoherent normalizations to demonstrate the presence of interference.

## Toy-MC closure validation

`tests/test_fit_closure.py` is the first full fitter closure test. It uses the same `pi- pi+ pi+` ordering and performs the complete validation chain:

```text
known truth parameters
    -> deterministic search for the accept-reject envelope maximum
    -> accept-reject toy generation from fresh phase-space batches
    -> several separated Minuit starting points
    -> independent MC normalization sample
    -> cached unbinned likelihood
    -> select the valid minimum with the lowest NLL
    -> pull and absolute-sanity checks against injected parameters
```

`ToyGenerator` uses accept-reject rather than categorical resampling from a finite pool. The native phase-space proposal is uniform in the unit square used to parametrize the Dalitz plot, so the proposal-to-Dalitz Jacobian returned as `PhaseSpaceSample.weights` enters the accept-reject score,

```text
score(u) = w_PS(u) |A(x(u))|^2.
```

The accept-reject envelope is no longer estimated from the maximum of a random pilot sample. Before generation, `ToyGenerator.estimate_maximum()` evaluates the score on a regular grid covering the full unit square, keeps several of the highest-score cells, and performs successive local refinements around those candidates. `envelope_safety` is then applied only as a safety margin above this deterministically located maximum. If a later generation batch nevertheless exceeds the envelope, the generator enlarges it and restarts the accepted sample so no event is retained with an inconsistent acceptance probability.

`pool_size` remains temporarily in the `ToyGenerator` constructor for backwards compatibility with existing examples, but it no longer controls envelope estimation.

The reference rho coefficient is fixed to remove the arbitrary global scale and phase. The test floats the `f0` and non-resonant magnitudes and phases and compares the fitted values with the injected truth, including wrapped phase differences. Because coherent amplitudes can contain separated local minima in phase space, closure validation does not rely on a single arbitrary Minuit start: the same cached objective is minimized from several starts, including one close to the injected point, and the valid minimum with the lowest NLL is selected. `Minimizer.fit(start_values=...)` provides these explicit starts without rebuilding the amplitude or normalization cache.

The closure test also compares `NLL(truth)` with the best fitted NLL. A finite toy may prefer a nearby parameter point, but a large separation is treated as evidence that toy generation or the fitted probability model is still inconsistent.

Closure is evaluated primarily through pulls using HESSE uncertainties, while broad absolute limits guard against wrong local minima or pathological error estimates.

For likelihood minimization, `Minimizer` uses the Minuit NLL convention `errordef=0.5` by default, and each `Parameter.step` is propagated to the corresponding Minuit initial error/step size.

Toy generation and fit normalization intentionally use independent Monte Carlo samples. A separate deterministic expected-NLL/Asimov regression test verifies that the likelihood normalization is stationary at the injected parameters when toy fluctuations are removed.

An interactive version of the same validation is available in:

```text
notebooks/01_dplus_fit_closure.ipynb
```

The notebook includes the generated toy Dalitz plot, truth/start/fit parameter comparison with fit uncertainties, one-dimensional `s12`, `s13` and `s23` projections before and after minimization, and side-by-side two-dimensional toy/model densities before and after the fit. The model projections use the independent normalization MC sample weighted by the fitted amplitude, so they visualize the same probability model used in the likelihood.

Closure tests are intended to be mandatory validation for new important coefficient parameterisations and dynamical line-shape implementations.

## Numerical convention

The signal PDF is

```text
p_sig(x | theta) = epsilon(x) |A(x; theta)|^2
                   ------------------------------------
                   integral epsilon(x) |A(x; theta)|^2 dPhi
```

For a linear model `A = sum_i c_i A_i`, the cached normalization matrix is

```text
M_ij = integral epsilon(x) A_i*(x) A_j(x) dPhi,
I    = c^dagger M c.
```

## QRules and sub-threshold resonances

`ReactionBuilder` exposes QRules' `mass_conservation_factor`. Set it to `None` when mass conservation should not reject intermediate states, for example for a sub-threshold resonance.

## Installation

```bash
git clone https://github.com/JuanBSLeite/DalitzPlotFitter.git
cd DalitzPlotFitter
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

For amplitude fits, double precision is recommended:

```python
from dalitzplotfitter import enable_x64
enable_x64()
```

## Current status

The package now contains CP-aware coefficient sets, fit-aware coefficient parameters, coherent external amplitude coefficients, cache-aware amplitude evaluation, native JAX three-body Dalitz kinematics and four-momentum reconstruction, QRules/AmpForm model building, TensorWaves/JAX compilation and symmetrized kinematic transformation, fixed-sample Monte Carlo normalization, deterministic-envelope accept-reject toy generation, efficiency/background interfaces, likelihood scaffolding, a Minuit/JAX bridge, fit/interference fractions and CP observables.

The current validation milestone is the full `D+ -> pi- pi+ pi+` toy-MC coefficient closure test. The next closure milestone is floating one or more dynamical parameters such as resonance mass or width while verifying selective cache invalidation and parameter recovery.
