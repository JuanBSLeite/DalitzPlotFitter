# DalitzPlotFitter

DalitzPlotFitter is a Python package under development for unbinned amplitude fits of three-body decays. It is built around **QRules**, **AmpForm**, **TensorWaves/JAX** and **mplhep**.

The project deliberately uses one numerical backend: **JAX**. JAX is an internal implementation choice and is not included in public class names.

## Design goals

- construct decay chains and validate quantum numbers with QRules;
- formulate symbolic amplitudes with AmpForm;
- compile numerical amplitudes to JAX through TensorWaves;
- perform deterministic Monte Carlo normalization on a fixed three-body phase-space sample generated natively with JAX;
- optionally include efficiency in the signal PDF;
- model background with analytic/JAX callables or Dalitz histograms;
- support signal-only weighted likelihoods with sWeights;
- support simultaneous particle/antiparticle fits with CP violation;
- calculate fit fractions, interference fractions and CP-asymmetry observables;
- use iminuit while JAX evaluates the NLL and gradients;
- provide HEP-style diagnostics and Dalitz plots with mplhep.

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

The same phase-space sample should be reused throughout a fit, making Monte Carlo normalization deterministic.

## First physical model: D+ -> pi+ pi+ pi-

The first integration benchmark is the symmetrized `D+ -> pi+ pi+ pi-` amplitude through `rho(770)0`. It matches the channel used in the AmpForm symmetrization documentation: QRules keeps one indistinguishable quantum-state transition and AmpForm restores the two kinematically distinct `pi+ pi-` pairings inside the amplitude.

Run it with:

```bash
python examples/01_dplus_rho.py
```

The example performs the complete chain

```text
QRules reaction
    -> AmpForm symbolic model
    -> TensorWaves/JAX compiled intensity
    -> native JAX three-body phase space
    -> AmpForm kinematic transformation
    -> intensity evaluation
    -> fixed-sample Monte Carlo normalization
```

`CompiledModel` exposes the numerical model as a pure `model(data, parameters)` function instead of relying on hidden mutable parameter updates. This is the interface that will be used by JAX autodiff and the likelihood fitter.

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

The package now contains CP-aware coefficient sets, native JAX three-body Dalitz kinematics and four-momentum reconstruction, QRules/AmpForm model building, TensorWaves/JAX compilation and kinematic transformation, fixed-sample Monte Carlo normalization, efficiency/background interfaces, likelihood scaffolding, a Minuit/JAX bridge, fit/interference fractions and CP observables. The next physics milestone is extending the D+ reference model beyond the rho-only benchmark to multiple resonances and a non-resonant amplitude, then fitting generated pseudo-data.
