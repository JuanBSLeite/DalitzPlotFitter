# DalitzPlotFitter

DalitzPlotFitter is a Python package under development for unbinned amplitude fits of three-body decays. The numerical pipeline is JAX end to end: phase-space generation, kinematics, amplitudes, normalization, likelihoods and gradients all run on the active JAX device. `iminuit` performs minimization, `particle` supplies standard particle properties, and `uproot` provides ROOT-file input without requiring PyROOT.

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

## ROOT input with uproot

ROOT files are supported directly through `uproot`, with no PyROOT dependency.

A fit sample can be read from a TTree with

```python
from dalitzplotfitter import read_phase_space_sample

data = read_phase_space_sample(
    "data.root",
    "DecayTree",
    s12="S12",
    s13="S13",
    s23="S23",
    weight="eventWeight",  # optional
)
```

For arbitrary observables and branch renaming use `read_root_tree`. Optional `cut`, `entry_start` and `entry_stop` arguments are forwarded to uproot.

ROOT TH2 maps can be converted directly into the package histogram models:

```python
from dalitzplotfitter import (
    histogram_background_from_root,
    histogram_efficiency_from_root,
)

efficiency = histogram_efficiency_from_root(
    "maps.root",
    "efficiency_s13_s23",
    x_variable="s13",
    y_variable="s23",
)

background = histogram_background_from_root(
    "maps.root",
    "background_s13_s23",
    x_variable="s13",
    y_variable="s23",
)
```

See `docs/root_io.md` for details, including optional four-momentum branches.

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

The mass-plane Gauss--Legendre implementation constructs a tensor-product rule in `m13` and `m23`, keeps the nodes inside the physical Dalitz boundary and includes the Jacobian `4*m13*m23`. The resulting sample is used unchanged by `SignalPDF` and `PreparedAmplitudeCache`, so direct and matrix normalizations share the same quadrature points and weights.

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

## Component normalization convention

Every dynamical component is normalized before applying its complex coefficient:

```text
integral dPhi |F_j|^2 = 1
```

Detector efficiency is excluded from the individual component normalization and enters only the total PDF normalization.

## Architecture

```text
ROOT TTree / arrays / generated sample
        -> PhaseSpaceSample
        -> DecayChannel + amplitude-component declarations
        -> pure-JAX kinematics
        -> resonance lineshape
        -> Blatt-Weisskopf factors
        -> angular factor
        -> automatic identical-particle symmetrization
        -> deterministic Gauss-Legendre or Square-Dalitz normalization
        -> coherent amplitude
        -> optional ROOT/array efficiency maps
        -> optional veto / SCF / background mixture
        -> optional factorized discriminating-variable PDFs
        -> optional external Gaussian constraints
        -> JAX NLL + automatic gradients
        -> iminuit
```

## Additional discriminating variables

Basic observables beyond the Dalitz plot can be added with factorized PDFs, for example reconstructed mass or a BDT output, using `FactorizedDensity`, `Gaussian1D`, `Exponential1D` and `Histogram1D`.

The factorized approximation is

```text
P(DP, x1, x2, ...) = P(DP) P(x1) P(x2) ...
```

for each signal or background category.

## External constraints

Gaussian external measurements can be added directly to any likelihood with `GaussianConstraint` and `ConstrainedNLL`.

## Phase-space Monte Carlo is for toys only

`PhaseSpaceMC` remains available for event/proposal generation. It implements three-body Lorentz-invariant phase space directly in JAX and returns the corresponding phase-space importance weight.

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
- `notebooks/09_b2kpipi_veto_maps.ipynb`: Laura++-style mass-window and functional veto maps applied consistently to data, signal and background normalization;
- `notebooks/10_b2kpipi_discriminating_variables.ipynb`: joint Dalitz + reconstructed-mass + BDT fit with factorized PDFs and mass/BDT projections;
- `notebooks/11_b2kpipi_gaussian_constraints.ipynb`: external Gaussian constraint on a fit parameter, including an NLL scan showing the effect of the constraint;
- `notebooks/12_b2kpipi_scf_with_veto.ipynb`: SCF migration combined with a veto applied in reconstructed Dalitz coordinates;
- `notebooks/13_b2kpipi_root_tree_input.ipynb`: synthetic ROOT TTree read with uproot and used directly as amplitude-fit input, with Dalitz and fitted projections;
- `notebooks/14_b2kpipi_root_hist_eff_background.ipynb`: ROOT TH2 efficiency and background maps loaded into `HistogramEfficiency` and `HistogramBackground`, with map and projection plots.

The B-to-Kpipi examples consistently use

```text
s13 = m^2(K+ pi-)
s23 = m^2(pi+ pi-)
```

for the particle ordering `(K+, pi+, pi-)`.

## Fit fractions

Fit fractions are evaluated with the same component convention and normalization matrix used by the model. The default gives physical fractions without detector efficiency; supplying an efficiency gives acceptance-weighted fractions explicitly.

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
