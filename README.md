# DalitzPlotFitter

DalitzPlotFitter is a Python package under development for unbinned amplitude fits of three-body decays. JAX evaluates amplitudes, likelihoods and gradients, iminuit performs minimization, `phasespace` supplies weighted Monte Carlo, and `particle` supplies standard particle properties.

Laura++ is one of the main physics references used to define and validate resonance, barrier-factor and angular conventions, but implementation classes use neutral names rather than backend/reference-specific names.

## High-level API

The preferred interface is channel driven:

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

normalization_mc = model.generate_phase_space(1_000_000, seed=2027)
pdf = model.pdf(normalization_mc)
```

`DecayChannel` resolves the parent and daughter masses from the `particle` package. `Resonance` resolves resonance mass, width and spin from `particle` unless an analysis-specific override is supplied.

This means users normally do **not** pass parent mass, daughter masses or bachelor mass to a lineshape/amplitude constructor.

Historical or analysis-specific values are explicit overrides, for example:

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

## Architecture

```text
DecayChannel + amplitude-component declarations
        -> particle masses / widths / spins
        -> phasespace weighted MC
        -> four-momenta and Dalitz invariants
        -> resonance lineshape
        -> Blatt-Weisskopf factors
        -> angular factor
        -> automatic identical-particle symmetrization
        -> RealImag coefficient
        -> coherent amplitude
        -> normalized SignalPDF
        -> JAX NLL + iminuit
```

The low-level complete resonance object is `ResonanceAmplitude`. The relativistic Breit-Wigner is exposed as `relativistic_breit_wigner`; neither the lineshape nor amplitude class carries a Laura++ or angular-formalism name.

## Angular convention

The current angular implementation uses the covariant formalism. Event-wise quantities are calculated directly from four-momenta:

```text
p*          bachelor momentum in the parent rest frame
p           bachelor momentum in the resonance rest frame
q           resonance-daughter momentum in the resonance rest frame
cos(theta)  angle between the selected daughter and bachelor in that frame
```

The detailed formulas and validation references are documented in `docs/lineshapes.md`.

## Automatic identical-particle symmetrization

For

```text
D+ -> pi- pi+ pi+
```

with ordering

```text
p1 = pi-
p2 = pi+_1
p3 = pi+_2
```

a resonance declared once with `pair=(0, 1)` sees that daughters 2 and 3 are identical and automatically evaluates

```text
F = F[(12)3] + F[(13)2].
```

There is no separate Bose-symmetrization wrapper. Constant non-resonant terms are not duplicated.

## Weighted phase-space Monte Carlo

`DecayModel.generate_phase_space()` uses `phasespace` with raw weights (`normalize_weights=False`). TensorFlow is confined to generation inside `phasespace`; generated arrays are immediately converted to the internal JAX representation.

For Monte Carlo normalization,

```text
N(theta) proportional to sum_k w_PS(k) |A(x_k; theta)|^2.
```

Unweighted pseudo-data are produced from a larger weighted pool using

```text
w_target(k) = w_PS(k) |A(x_k; theta_gen)|^2
```

followed by `weighted_resample()`.

## E791 Fit 2 generation example

`notebooks/01_e791_dplus_fit2_generation.ipynb` gives a generation example based on Fit 2 of E791, arXiv:hep-ex/0007028v2. The notebook uses the high-level `DecayChannel` / `DecayModel` API, `particle` for standard particle properties, explicit paper-specific overrides where needed, a 1,000,000-event weighted candidate pool and 100,000 resampled pseudo-data events, with Dalitz and projection plots.

## Coefficients

The supported complex coefficient parameterization is

```text
c = x + i y
```

through `RealImag`. `x` and `y` may be constants or fit `Parameter` objects.

For closure of a fitted coordinate,

```text
abs((value_gen - value_fit) / sigma_fit) < 1
```

using HESSE uncertainties with `errordef=0.5`.

## Current roadmap

1. relativistic Breit-Wigner — implemented;
2. covariant angular formalism — implemented for `L=0..4`;
3. weighted `phasespace` MC — implemented;
4. `particle`-driven decay/channel model — implemented;
5. automatic identical-particle symmetrization — implemented;
6. E791 Fit 2 generation notebook — implemented;
7. end-to-end 100k/1M closure on the new path — next;
8. Gounaris-Sakurai;
9. Flatte;
10. LASS and K-matrix.

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

The project uses standard amplitude-analysis literature as validation references. Laura++ is currently a principal reference for resonance, Blatt-Weisskopf and covariant angular conventions:

J. Back et al., *Laura++: a Dalitz plot fitter*, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.
