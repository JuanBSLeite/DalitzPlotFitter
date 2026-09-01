# DalitzPlotFitter

DalitzPlotFitter is a Python package under development for unbinned amplitude fits of three-body decays. The numerical pipeline is JAX end to end: phase-space generation, kinematics, amplitudes, normalization, likelihoods and gradients all run on the active JAX device. `iminuit` performs minimization and `particle` supplies standard particle properties.

There is no TensorFlow dependency or mixed TensorFlow/JAX numerical path.

Laura++ is one of the main physics references used to define and validate resonance, barrier-factor and angular conventions, but implementation classes use neutral names rather than backend/reference-specific names.

## High-level API

```python
from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    NonResonant,
    RealImag,
    Resonance,
)

channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))

model = DecayModel(
    channel,
    [
        Resonance("rho(770)0", pair=(0, 1), coefficient=RealImag(1.0, 0.0)),
        NonResonant(RealImag(0.2, -0.1)),
    ],
)

pdf = model.pdf()
```

`DecayChannel` resolves the parent and daughter masses from `particle`. `Resonance` resolves resonance mass, width and spin from `particle` unless an analysis-specific override is supplied.

Historical or analysis-specific values can be supplied explicitly:

```python
Resonance(
    "sigma",
    pair=(0, 1),
    coefficient=RealImag(x, y),
    mass=0.478,
    width=0.324,
    spin=0,
)
```

## Normalization

Amplitude and PDF normalization use deterministic quadrature. The default is
the equal-area `DalitzGrid`; a Laura++-style Gauss--Legendre method is also
available.

```python
from dalitzplotfitter import DalitzGrid

norm = DalitzGrid(
    channel.parent_mass,
    channel.daughter_masses,
    resolution=1000,
).sample()
cache = model.prepare_cache(data, norm)
```

A resolution `N` gives exactly `N**2` physical integration points. The default `DecayModel` normalization uses

```python
model = DecayModel(
    channel,
    components,
    normalize_components=True,
    normalization_resolution=1000,
)
```

so the default normalization support contains exactly one million deterministic grid points. It is created lazily and reused for the lifetime of the model.

There is no Monte Carlo normalization path in the supported API.

To use the Laura++ prescription in both component and total-PDF normalization:

```python
model = DecayModel(
    channel,
    components,
    normalization_method="laura",
    normalization_bin_width=0.005,  # GeV; Laura++ default is 5 MeV
)
```

This constructs a tensor-product Gauss--Legendre rule in `m13` and `m23`,
keeps the nodes inside the physical Dalitz boundary and includes the Jacobian
`4*m13*m23`. The resulting sample is used unchanged by `SignalPDF` and
`PreparedAmplitudeCache`, so direct and matrix normalizations share the same
quadrature points and weights. Explicit `normalization_order_m13` and
`normalization_order_m23` values can be supplied for convergence studies.

Laura++'s additional sub-grid treatment for resonances narrower than 20 MeV is
not enabled by this base method yet; use the existing amplitude-aware adaptive
grids for such models and validate every normalization-matrix element.

## Component normalization convention

Every dynamical component is normalized before applying its complex coefficient:

```text
integral dPhi |F_j|^2 = 1
```

This keeps coefficient scales stable when dynamical parameters such as resonance masses or widths float. Detector efficiency is excluded from the individual component normalization and enters only the total PDF normalization.

## Architecture

```text
DecayChannel + amplitude-component declarations
        -> particle masses / widths / spins
        -> pure-JAX kinematics
        -> resonance lineshape
        -> Blatt-Weisskopf factors
        -> angular factor
        -> automatic identical-particle symmetrization
        -> deterministic DalitzGrid component normalization
        -> RealImag coefficient
        -> coherent amplitude
        -> deterministic DalitzGrid PDF normalization
        -> JAX NLL + automatic gradients
        -> iminuit
```

## Phase-space Monte Carlo is for toys only

`PhaseSpaceMC` remains available for event/proposal generation. It implements three-body Lorentz-invariant phase space directly in JAX and returns the corresponding phase-space importance weight.

Toy generation can use a weighted pool:

```text
w_target(k) = w_PS(k) |A(x_k; theta_gen)|^2
```

followed by `weighted_resample()`.

`PhaseSpaceMC` is **not** used for amplitude or PDF normalization.

## Deterministic Dalitz grid

`DalitzGrid` constructs an equal-area grid in transformed Dalitz coordinates. A resolution `N` gives exactly `N**2` physical integration points with constant quadrature weights. Details are in `docs/dalitz_grid_integration.md`.

## E791 examples

Current validation examples use deterministic grid normalization:

- `notebooks/02_fit_dynamic_parameters.ipynb`: E791 coefficient closure;
- `notebooks/03_lineshape_parameter_diagnostics.ipynb`: lineshape diagnostics;
- `notebooks/04_normalization_grid_diagnostics.ipynb`: grid-convergence and normalization diagnostics;
- `notebooks/07_e791_rho1450_mass_width_closure.ipynb`: coefficients plus `rho(1450)` mass and width.
- `notebooks/15_laura_e791_integration_validation.ipynb`: Laura++-style versus
  equal-area normalization using the complete seven-component E791 Fit 2 model,
  including matrix-element convergence and direct/cache closure.
- `notebooks/16_laura_e791_genfit.ipynb`: end-to-end E791 Fit 2 pseudo-data
  generation and Cartesian-coefficient fit with Laura++-style normalization.
- `notebooks/17_laura_e791_genfit_ensemble.ipynb`: repeated E791 Fit 2
  pseudoexperiment generation and fitting, with bias compatibility, fit-validity
  and pull mean/width diagnostics.

## Coefficients

The supported complex coefficient parameterization is

```text
c = x + i y
```

through `RealImag`. `x` and `y` may be constants or fit `Parameter` objects.

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

Double precision is recommended:

```python
from dalitzplotfitter import enable_x64
enable_x64()
```

## Physics references

J. Back et al., *Laura++: a Dalitz plot fitter*, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.
