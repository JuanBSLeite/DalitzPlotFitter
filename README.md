# DalitzPlotFitter

DalitzPlotFitter is a Python package under development for unbinned amplitude fits of three-body decays. The numerical pipeline is JAX end to end: phase-space generation, kinematics, amplitudes, normalization, likelihoods and gradients all run on the active JAX device. `iminuit` performs minimization, `particle` supplies standard particle properties, and `uproot` provides ROOT-file input/output without requiring PyROOT.

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

## User-friendly toy generation

There are two public unweighted toy-generation methods:

```text
inverse-transform
accept-reject
```

Inverse transform is the default because it is substantially faster for realistic amplitude models:

```python
from dalitzplotfitter import generate_toy

toy = generate_toy(
    model,
    1_000_000,
    parameters=fit_values,
    inverse_resolution=1024,
    seed=2,
)
```

The inverse method uses a numerical Rosenblatt transform on the physical conventional Dalitz plane. It first samples the marginal distribution and then the conditional distribution, including the exact coordinate Jacobian of the physical Dalitz boundary.

`accept-reject` remains available explicitly as an independent reference and validation sampler:

```python
toy_reference = generate_toy(
    model,
    50_000,
    parameters=fit_values,
    method="accept-reject",
    seed=1,
)
```

For repeated toys, prepare the inverse CDFs once:

```python
from dalitzplotfitter import prepare_inverse_toy_generator

prepared = prepare_inverse_toy_generator(
    model,
    parameters=fit_values,
    efficiency=efficiency,
    veto=veto,
    resolution=1024,
)

toy1 = prepared.generate(100_000, seed=10)
toy2 = prepared.generate(100_000, seed=11)
toy3 = prepared.generate(1_000_000, seed=12)
```

Efficiency, vetoes and backgrounds can be included directly in either public method. Toy samples can also be written directly to ROOT:

```python
toy = generate_toy(
    model,
    100_000,
    parameters=fit_values,
    seed=3,
    output_root="toy.root",
)
```

For simultaneous direct-CP pseudoexperiments, `generate_cp_toy` computes the accepted B+/B- charge split from the model integrals automatically. When ROOT output is requested, both charges are written to one TTree with `charge=+1` for B+ and `charge=-1` for B-:

```python
plus_toy, minus_toy = generate_cp_toy(
    plus_model,
    minus_model,
    50_000,
    parameters=fit_values,
    seed=4,
    output_root="cp_toy.root",
)
```

See `docs/toy_generation.md` and `notebooks/19_toy_root_output.ipynb`.

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

One-dimensional fitted projections use a separate weighted phase-space rendering sample so arbitrary histogram bins remain smooth. This rendering sample does not replace the deterministic quadrature used for likelihood normalization or fit fractions.

## ROOT input/output with uproot

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

Amplitude and PDF normalization use deterministic quadrature. The supported methods are `gauss-legendre`, `square-dalitz`, and `adaptive`.

The `adaptive` method follows the Laura++ integration strategy for explicit narrow resonances. Resonances with nominal width at or below 20 MeV are refined locally in a mass window `m0 ± 5*Gamma`; the default fine target spacing is `Gamma/100`, while the rest of the conventional Dalitz plane keeps the 5 MeV target spacing. Identical-particle symmetrisation is included when locating narrow bands, and overlapping bands automatically use the finest requested spacing. If a narrow band lies on the diagonal `m12` axis of the conventional `(m13,m23)` plane, the integration switches to a full Square-Dalitz grid, matching the Laura++ prescription.

```python
model = DecayModel(
    channel,
    components,
    normalization_method="adaptive",
    normalization_bin_width=0.005,       # 5 MeV outside narrow bands
    normalization_narrow_width=0.020,    # Gamma <= 20 MeV is narrow
    normalization_narrow_window=5.0,     # m0 ± 5 Gamma
    normalization_binning_factor=100.0,  # fine spacing Gamma/100
    normalization_resolution=1000,       # SDP fallback resolution
)
```

The integration scheme is fixed from the declared/initial resonance masses and widths, as in Laura++. It can be inspected through `model.adaptive_normalization_scheme`.

The original fixed grids remain available explicitly:

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

Every dynamical component is normalized by default before applying its complex coefficient:

```text
integral dPhi |F_j|^2 = 1
```

Detector efficiency is excluded from individual component normalization and enters only total PDF normalization.

An individual component can keep its raw dynamical scale while all other
components retain the model default:

```python
Resonance(
    "S_wave_QMI",
    pair=(0, 2),
    coefficient=coefficient,
    lineshape=qmi,
    normalize_component=False,
)
```

This disables only the unit-integral rescaling of that component. It remains
included in the coherent amplitude, interference terms, and total PDF
normalization. The default `normalize_component=None` inherits
`DecayModel.normalize_components`.

## Detector-resolution convolution

A generic one-dimensional convolution layer is available for continuously smeared observables. The same relativistic resonance lineshape used by the amplitude model can be converted into an isolated normalized intensity and convolved with a Gaussian detector response:

```python
from dalitzplotfitter import (
    ConvolvedPDF1D,
    GaussianResolution1D,
    LineshapeIntensity1D,
    RelativisticBreitWigner,
)

true_mass = LineshapeIntensity1D.from_context(
    RelativisticBreitWigner(),
    context,
    quadrature_order=512,
)

reco_mass = ConvolvedPDF1D(
    true_mass,
    GaussianResolution1D(sigma=0.008),
    true_low=true_mass.low,
    true_high=true_mass.high,
    observed_low=true_mass.low,
    observed_high=true_mass.high,
    quadrature_order=192,
)
```

For interfering amplitudes, detector resolution must act on the full coherent intensity rather than on each component intensity independently. See `docs/convolution_resolution.md` and `notebooks/20_pdf_convolution_resolution.ipynb`.

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
        -> optional discriminating-variable PDFs / 1D resolution convolution
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

## Phase-space Monte Carlo

`PhaseSpaceMC` is used for accept-reject proposal/event generation and for weighted rendering samples used by smooth fitted projections. It is **not** used for amplitude/PDF normalization or fit fractions, which remain deterministic. The default inverse-transform toy path instead samples the physical Dalitz plane from prepared inverse CDFs.

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
- `notebooks/17_b2kpipi_cp_user_friendly.ipynb`: concise `CPFitSession` workflow and charge-separated fitted projections;
- `notebooks/18_user_friendly_toy_generation.ipynb`: signal/background and CP pseudo-data generation;
- `notebooks/19_toy_root_output.ipynb`: non-CP and CP toy generation with ROOT TTree output;
- `notebooks/20_pdf_convolution_resolution.ipynb`: relativistic Breit-Wigner intensity convolved with Gaussian detector resolution.

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
