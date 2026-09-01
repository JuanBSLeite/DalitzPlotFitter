# DalitzPlotFitter

DalitzPlotFitter is a Python package under development for unbinned amplitude fits of three-body decays. The numerical pipeline is JAX end to end: phase-space generation, kinematics, amplitudes, normalization, likelihoods and gradients all run on the active JAX device. `iminuit` performs minimization, `particle` supplies standard particle properties, and `uproot` provides ROOT-file input without requiring PyROOT.

There is no TensorFlow dependency or mixed TensorFlow/JAX numerical path.

Laura++ is one of the main physics references used to define and validate resonance, barrier-factor and angular conventions, but implementation classes use neutral names rather than backend/reference-specific names.

## High-level API

```python
from dalitzplotfitter import DecayChannel, DecayModel, NonResonant, RealImag, Resonance

channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
model = DecayModel(
    channel,
    [
        Resonance("rho(770)0", pair=(0, 1), coefficient=RealImag(1.0, 0.0)),
        NonResonant(RealImag(0.2, -0.1)),
    ],
)
```

For common fits, `FitSession` composes the PDF, likelihood, backgrounds, constraints and minimizer automatically:

```python
from dalitzplotfitter import FitSession

session = FitSession(model, data)
result = session.fit(simplex=True)
session.report(result)
session.plot_projection(result, "s13")
```

A ROOT-file workflow can be reduced to:

```python
session = FitSession.from_root(
    model,
    "data.root",
    "DecayTree",
    s12="S12",
    s13="S13",
    s23="S23",
)
result = session.fit()
```

Background shapes can be supplied through `BackgroundSpec`; their deterministic Dalitz normalization is computed automatically.

## Simultaneous CP fits

`CPFitSession` removes the manual construction of the two prepared caches, joint CP likelihood and minimizer:

```python
from dalitzplotfitter import CPFitSession

session = CPFitSession(
    plus_model,
    minus_model,
    plus_data,
    minus_data,
)
result = session.fit()
session.report(result)
session.plot_projection(result, "s13")
```

Shared parameters are collected only once. Efficiency, vetoes and `CPBackgroundSpec` categories are folded into the same joint charge-Dalitz normalization convention used by `CPJointNLL`.

Charge-separated projection plots use one common global normalization, so the B+ and B- projections do not hide an integrated charge asymmetry through independent rescaling.

See `docs/user_friendly_api.md`.

## Plot helpers

```python
from dalitzplotfitter import plot_dalitz, plot_square_dalitz

plot_dalitz(data, x="s13", y="s23")
plot_square_dalitz(
    data,
    mother_mass=channel.parent_mass,
    masses=channel.daughter_masses,
    pair=(0, 2),
)
```

## ROOT input with uproot

ROOT files are supported directly through `uproot`, with no PyROOT dependency.

```python
from dalitzplotfitter import read_phase_space_sample

data = read_phase_space_sample(
    "data.root", "DecayTree",
    s12="S12", s13="S13", s23="S23",
    weight="eventWeight",
)
```

ROOT TH2 maps in ordinary Dalitz variables can be loaded with `histogram_efficiency_from_root` and `histogram_background_from_root`.

For B-decay analyses, ROOT TH2 maps defined directly in Square-Dalitz coordinates are also supported:

```python
from dalitzplotfitter import (
    square_dalitz_background_from_root,
    square_dalitz_efficiency_from_root,
)

kwargs = dict(
    mother_mass=channel.parent_mass,
    masses=channel.daughter_masses,
    pair=(0, 2),
)

efficiency = square_dalitz_efficiency_from_root(
    "maps.root", "efficiency_sdp", **kwargs
)
background = square_dalitz_background_from_root(
    "maps.root", "background_sdp", **kwargs
)
```

The TH2 axes are interpreted as `(m', theta')`. During PDF evaluation the fitter converts `(s12,s13,s23)` internally to Square-Dalitz coordinates before the bin lookup. The ordered `pair` must match the convention used to build the external maps.

See `docs/root_io.md` for details.

## Normalization

Amplitude and PDF normalization use deterministic quadrature. The supported methods are `gauss-legendre` and `square-dalitz`.

```python
model = DecayModel(
    channel,
    components,
    normalization_method="square-dalitz",
    normalization_resolution=1000,
    normalization_pair=(0, 1),
)
```

Square-Dalitz histogram values are scalar efficiency/background values; they do not receive an extra Jacobian. The coordinate-transformation Jacobian belongs to the integration measure and is already carried by `SquareDalitzGrid` normalization weights.

## Component normalization convention

Every dynamical component is normalized before applying its complex coefficient:

```text
integral dPhi |F_j|^2 = 1
```

Detector efficiency is excluded from individual component normalization and enters only total PDF normalization.

## Architecture

```text
ROOT TTree / arrays / generated sample
        -> PhaseSpaceSample
        -> DecayChannel + amplitude components
        -> pure-JAX kinematics and dynamics
        -> deterministic Dalitz / Square-Dalitz normalization
        -> coherent amplitude
        -> optional ordinary-Dalitz or Square-Dalitz efficiency/background maps
        -> optional veto / SCF / multiple backgrounds
        -> optional discriminating-variable PDFs
        -> optional Gaussian constraints
        -> FitSession / CPFitSession convenience layer (optional)
        -> JAX NLL + automatic gradients
        -> iminuit
```

The low-level `SignalPDF`, `PreparedAmplitudeCache`, likelihood and `Minimizer` classes remain public for advanced analyses and numerical validation.

## Additional discriminating variables

Basic observables beyond the Dalitz plot can be added with factorized PDFs using `FactorizedDensity`, `Gaussian1D`, `Exponential1D` and `Histogram1D`.

## External constraints

Gaussian external measurements can be added with `GaussianConstraint` and `ConstrainedNLL`, or attached directly to a fit session.

## Phase-space Monte Carlo is for toys only

`PhaseSpaceMC` remains available for event/proposal generation and is not used for amplitude or PDF normalization.

## Tutorial notebooks

The repository contains a progressive set of examples:

- `notebooks/01_e791_toy_fit.ipynb`: E791 signal toy generation and fit;
- `notebooks/02_e791_efficiency_background_fit.ipynb`: E791 efficiency/background fit;
- `notebooks/03_b2kpipi_toy_fit.ipynb`: non-CP `B+ -> K+ pi+ pi-` toy fit;
- `notebooks/04_b2kpipi_efficiency_background_fit.ipynb`: B efficiency/background fit;
- `notebooks/05_b2kpipi_cp_toy_fit.ipynb`: simultaneous direct-CP signal fit;
- `notebooks/06_b2kpipi_cp_efficiency_background_fit.ipynb`: CP fit with efficiency/background;
- `notebooks/07_b2kpipi_scf_migration.ipynb`: SCF migration;
- `notebooks/08_b2kpipi_multiple_backgrounds.ipynb`: arbitrary multiple backgrounds;
- `notebooks/09_b2kpipi_veto_maps.ipynb`: veto maps;
- `notebooks/10_b2kpipi_discriminating_variables.ipynb`: Dalitz + mass + BDT;
- `notebooks/11_b2kpipi_gaussian_constraints.ipynb`: Gaussian constraints;
- `notebooks/12_b2kpipi_scf_with_veto.ipynb`: SCF + reconstructed-space veto;
- `notebooks/13_b2kpipi_root_tree_input.ipynb`: ROOT TTree input;
- `notebooks/14_b2kpipi_root_hist_eff_background.ipynb`: ROOT TH2 maps in ordinary Dalitz coordinates;
- `notebooks/15_b2kpipi_square_dalitz_eff_background.ipynb`: ROOT TH2 efficiency/background maps in `(m', theta')`;
- `notebooks/16_user_friendly_quickstart.ipynb`: concise non-CP `FitSession` workflow;
- `notebooks/17_b2kpipi_cp_user_friendly.ipynb`: concise `CPFitSession` workflow, automatic report and charge-separated fitted projections.

The B-to-Kpipi examples consistently use

```text
s13 = m^2(K+ pi-)
s23 = m^2(pi+ pi-)
```

for particle ordering `(K+, pi+, pi-)`.

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
