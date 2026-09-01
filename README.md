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

Amplitude and PDF normalization use deterministic quadrature. The supported
methods are `gauss-legendre` (default) and `square-dalitz`.

```python
from dalitzplotfitter import DalitzGaussLegendreGrid

norm = DalitzGaussLegendreGrid(
    channel.parent_mass,
    channel.daughter_masses,
    bin_width=0.005,
).sample()
cache = model.prepare_cache(data, norm)
```

The default `DecayModel` normalization uses

```python
model = DecayModel(
    channel,
    components,
    normalize_components=True,
    normalization_method="gauss-legendre",
    normalization_bin_width=0.005,
)
```

The quadrature sample is created lazily and reused for the lifetime of the model.

There is no Monte Carlo normalization path in the supported API.

The mass-plane Gauss--Legendre implementation uses

```python
model = DecayModel(
    channel,
    components,
    normalization_method="gauss-legendre",
    normalization_bin_width=0.005,  # GeV; Laura++ default is 5 MeV
)
```

It constructs a tensor-product rule in `m13` and `m23`,
keeps the nodes inside the physical Dalitz boundary and includes the Jacobian
`4*m13*m23`. The resulting sample is used unchanged by `SignalPDF` and
`PreparedAmplitudeCache`, so direct and matrix normalizations share the same
quadrature points and weights. Explicit `normalization_order_m13` and
`normalization_order_m23` values can be supplied for convergence studies.

This numerical prescription is the same base normalization method used by
Laura++. The implementation name describes the quadrature and does not encode
the name of the reference package.

Square-Dalitz normalization is selected with

```python
model = DecayModel(
    channel,
    components,
    normalization_method="square-dalitz",
    normalization_resolution=1000,
    normalization_pair=(0, 1),
)
```

The former equal-area and adaptive normalization methods are not part of the
supported first-version API.

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
        -> deterministic Gauss-Legendre or Square-Dalitz normalization
        -> RealImag coefficient
        -> coherent amplitude
        -> the same quadrature for PDF normalization
        -> optional efficiency / veto / SCF / background mixture
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

## Tutorial notebooks

The repository contains a progressive set of end-to-end examples:

- `notebooks/01_e791_toy_fit.ipynb`: E791 signal toy generation and fit;
- `notebooks/02_e791_efficiency_background_fit.ipynb`: E791 fit with efficiency and background;
- `notebooks/03_b2kpipi_toy_fit.ipynb`: non-CP `B+ -> K+ pi+ pi-` signal toy and fit;
- `notebooks/04_b2kpipi_efficiency_background_fit.ipynb`: the same B channel with efficiency and background;
- `notebooks/05_b2kpipi_cp_toy_fit.ipynb`: simultaneous direct-CP signal-only fit;
- `notebooks/06_b2kpipi_cp_efficiency_background_fit.ipynb`: direct-CP fit with efficiency and background, including non-extended and extended likelihood usage;
- `notebooks/07_b2kpipi_scf_migration.ipynb`: Laura++-style SCF / misreconstructed-event migration matrix, CR+SCF decomposition and normalization conservation;
- `notebooks/08_b2kpipi_multiple_backgrounds.ipynb`: arbitrary multiple background categories, signal-fraction convention and extended per-category yields;
- `notebooks/09_b2kpipi_veto_maps.ipynb`: Laura++-style mass-window and functional veto maps applied consistently to data, signal and background normalization.

The B-to-Kpipi examples consistently use

```text
s13 = m^2(K+ pi-)
s23 = m^2(pi+ pi-)
```

for the particle ordering `(K+, pi+, pi-)`.

## Fit fractions

Fit fractions are evaluated with the same component convention and
normalization matrix used by the model:

```python
fractions = model.fit_fractions(fit_values)
model.print_fit_fractions(
    fit_values,
    include_interference=True,
)
```

The default gives physical fractions without detector efficiency. Supplying
`efficiency=efficiency_model` gives acceptance-weighted fractions explicitly.
Fit fractions need not sum to one because interference is coherent.

## Coefficients

The supported complex coefficient parameterization is

```text
c = x + i y
```

through `RealImag`. `x` and `y` may be constants or fit `Parameter` objects.

Direct-CP examples use `CPRealImag` with shared CP-even and CP-odd Cartesian parameters.

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
