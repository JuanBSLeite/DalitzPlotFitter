# DalitzPlotFitter

DalitzPlotFitter is a Python package for **unbinned amplitude fits of three-body decays**
("Dalitz plot analyses"), the technique used across flavour physics (LHCb, BaBar, Belle,
BESIII, E791, ...) to extract resonance parameters, branching fractions, CP asymmetries and
interference structure from a sample of reconstructed decays such as `B+ -> K+ pi+ pi-` or
`D+ -> pi- pi+ pi+`.

The whole numerical pipeline — phase-space generation, kinematics, amplitude dynamics,
normalization, likelihood and gradient evaluation — is written in **JAX**, end to end, on
`float64`/`complex128`. `iminuit` performs the final minimization against a JAX
`value_and_grad` objective; `particle` supplies standard particle properties; `uproot` provides
ROOT file input/output without needing PyROOT. There is no TensorFlow dependency anywhere.

Laura++ is the primary physics reference for resonance, barrier-factor, angular and
Square-Dalitz-Plot conventions; DalitzPlotFitter follows those conventions (and documents any
deliberate deviation) while using its own, backend-neutral class names.

## What it can do

- **Build an amplitude model** for a three-body decay from a coherent sum of resonances (plus a
  non-resonant term), each with a complex coefficient and a lineshape (relativistic
  Breit-Wigner, Gounaris-Sakurai, Flatte, LASS, a K-matrix, various pole and rescattering
  parametrizations, or a fully two-dimensional QMI amplitude).
- **Fit** that model to data with an unbinned maximum-likelihood fit, including detector
  efficiency, vetoed regions, self-cross-feed (SCF) migration, an arbitrary number of background
  categories, additional discriminating variables (mass, BDT output, ...) and Gaussian external
  constraints — any combination of these can be switched on independently.
- **Fit simultaneously for CP asymmetries**, treating the particle and antiparticle samples as
  one joint normalized likelihood rather than two independent fits, so the fit is directly
  sensitive to the integrated charge asymmetry.
- **Generate toy Monte Carlo** from a fitted or hypothesized model — signal, background, and
  simultaneous CP pseudo-experiments — either as an exact numerical inverse-transform (fast,
  default) or as a Laura++-style accept-reject sampler (used as an independent cross-check).
- **Read and write ROOT files** directly (via `uproot`), including loading efficiency/background
  maps defined as ROOT histograms in plain Dalitz or Square-Dalitz coordinates.
- **Handle decays with two identical final-state particles** automatically and correctly on the
  full, unfolded Dalitz plane, with optional folded views for making efficiency/background maps
  from limited statistics or for diagnostic plots.
- **Produce standard plots**: Dalitz plot, Square Dalitz plot, and smooth one-dimensional fitted
  projections with residuals/pulls.
- **Scale to large samples and floating-dynamics fits efficiently**, through a caching layer that
  avoids re-evaluating fixed parts of the amplitude and the normalization integral at every
  minimizer step.

## How you would use it

There are two ways to work with the package, and both stay available at the same time:

1. **The convenience layer** (`FitSession` for a single sample, `CPFitSession` for simultaneous
   CP fits) composes the PDF, likelihood, backgrounds, constraints and minimizer for you, and
   exposes `.fit()`, `.report()` and `.plot_projection()`. This is the recommended starting point
   for a standard analysis and is what the tutorial notebooks use throughout.
2. **The low-level public classes** (`SignalPDF`, `PreparedAmplitudeCache`, `MultiBackgroundNLL`,
   `CPJointNLL`, `Minimizer`, ...) that the convenience layer is built from remain fully public,
   for advanced setups, custom likelihoods, or numerical validation work that needs direct
   control over any stage of the pipeline.

In practice, using the package means: describe the decay and its resonances, wrap the model and
data in a `FitSession` (or `CPFitSession` for CP), call `.fit()`, then inspect the result with
`.report()` and `.plot_projection()`. Everything else — efficiency maps, backgrounds, vetoes,
constraints, ROOT I/O, toy generation — plugs into that same model/session without changing this
basic flow.

## Where to go next

- [`notebooks/tutorials/TUTORIALS.md`](notebooks/tutorials/TUTORIALS.md) — a self-contained,
  nine-part course, from phase space and a first fit through normalization, acceptance, floating
  dynamics, CP fits, ROOT I/O, QMI and goodness of fit. Start here.
- [`docs/catalog.md`](docs/catalog.md) — one line per public class/function, with a pointer to
  the doc or notebook that covers it in depth. Use it to answer "does something already do X" or
  "where is X" before searching the source.
- Focused docs under [`docs/`](docs/) cover each subsystem in depth: fitting, lineshapes, Monte
  Carlo integration/normalization, backgrounds and vetoes, CP coefficients, SCF, Square Dalitz
  Plot conventions, toy generation, discriminating variables and constraints, convolution/
  resolution, internal dynamics structure, performance/caching, and ROOT I/O.
- [`notebooks/`](notebooks) contains a progressive set of worked examples beyond the tutorial
  course — efficiency/background fits, multiple backgrounds, veto maps, SCF migration, Gaussian
  constraints, ROOT I/O, folded Dalitz plots for identical particles, resolution convolution, and
  more — plus `notebooks/benchmark/` (numeric reproductions of published analyses) and
  `notebooks/data_analyses/` (in-progress analyses using the public API).

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

Double precision must be enabled once before any numerical work, since the project deliberately
runs `float64`/`complex128` rather than JAX's default:

```python
from dalitzplotfitter import enable_x64
enable_x64()
```

## Physics references

J. Back et al., *Laura++: a Dalitz plot fitter*, Computer Physics Communications 231 (2018)
198-242, arXiv:1711.09854.

LHCb Collaboration, *Amplitude analysis of the D<sub>s</sub><sup>+</sup> → π<sup>-</sup>π<sup>+</sup>π<sup>+</sup>
decay*, Phys. Rev. D 99 (2019) 012011, arXiv:1811.08688.

LHCb Collaboration, *Amplitude analysis of the D<sup>+</sup> → π<sup>-</sup>π<sup>+</sup>π<sup>+</sup> decay*,
Phys. Rev. D 103 (2021) 092004, arXiv:2009.00025.

LHCb Collaboration, *Amplitude analysis of the B<sup>±</sup> → π<sup>±</sup>π<sup>±</sup>π<sup>∓</sup> decay*,
Phys. Rev. D 101 (2020) 012006, arXiv:1909.05212.
