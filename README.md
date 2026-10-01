<table>
<tr>
<td width="190"><img src="assets/logo.png" alt="Jax-PWA logo" width="170"></td>
<td>

# Jax-PWA

**JAX-native amplitude fitting for three-body decays, with initial four-body support.**

</td>
</tr>
</table>

[![tests](https://github.com/JuanBSLeite/Jax-PWA/actions/workflows/tests.yml/badge.svg)](https://github.com/JuanBSLeite/Jax-PWA/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)
![JAX](https://img.shields.io/badge/backend-JAX-orange)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JuanBSLeite/Jax-PWA/main?filepath=notebooks/tutorials)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuanBSLeite/Jax-PWA)

Jax-PWA is a Python package for **unbinned amplitude fits of three-body decays**
("Dalitz plot analyses"), the technique used across flavour physics (LHCb, BaBar, Belle,
BESIII, E791, ...) to extract resonance parameters, branching fractions, CP asymmetries and
interference structure from a sample of reconstructed decays such as `B+ -> K+ pi+ pi-` or
`D+ -> pi- pi+ pi+`.

Initial [four-body support](docs/four_body.md) adds invariant event coordinates,
weighted N-body phase space and LS-coupled pair/cascade chains for scalar external
particles, using the same JAX cache, normalization and `FitSession` minimization.
Start with the [four-body closure example](notebooks/examples/four_body_closure.py)
or the [BESIII-inspired D0 -> pi+ pi- pi+ pi- reproduction](notebooks/examples/besiii_d0_4pi_toy_reproduction.ipynb), which includes covariant
Zemach/Rarita-Schwinger tensors and a native helicity/LS comparison.

The whole numerical pipeline — phase-space generation, kinematics, amplitude dynamics,
normalization, likelihood and gradient evaluation — is written in **JAX**, end to end, on
`float64`/`complex128`. `iminuit` performs the final minimization against a JAX
`value_and_grad` objective; `particle` supplies standard particle properties; `uproot` provides
ROOT file input/output without needing PyROOT. There is no TensorFlow dependency anywhere.

Laura++ is the primary physics reference for resonance, barrier-factor, angular and
Square-Dalitz-Plot conventions; Jax-PWA follows those conventions (and documents any
deliberate deviation) while using its own, backend-neutral class names.

## What it can do

- **Build an amplitude model** for a three-body decay from a coherent sum of resonances (plus a
  non-resonant term), each with a complex coefficient and a lineshape (relativistic
  Breit-Wigner, Gounaris-Sakurai, Flatte, LASS, a K-matrix, various pole and rescattering
  parametrizations, or a fully two-dimensional QMI amplitude).
- **Build initial four-body amplitude models** for scalar external particles with physical
  weighted phase space, Lorentz-invariant event representations, reusable isobars, identical-
  particle symmetrization, and sequential `P -> R1 R2` or `P -> R a -> S b a` decay chains.
- **Fit** that model to data with an unbinned maximum-likelihood fit, including detector
  efficiency, vetoed regions, self-cross-feed (SCF) migration, an arbitrary number of background
  categories, additional discriminating variables (mass, BDT output, ...) and Gaussian external
  constraints — any combination of these can be switched on independently.
- **Fit simultaneously for CP asymmetries**, treating the particle and antiparticle samples as
  one joint normalized likelihood rather than two independent fits, so the fit is directly
  sensitive to the integrated charge asymmetry.
- **Fit tagged time-dependent neutral-meson amplitudes**, including coherent `D0`-`D0bar`
  mixing in `D0 -> KS pi+ pi-`, JAX gradients, time acceptance, optional Gaussian time
  resolution, multiple backgrounds, fraction or extended-yield fits, and signal/background
  projections through `NeutralMesonMixing`, `TimeDependentDalitzNLL`, and
  `TimeDependentFitSession`. The convention follows the Belle model (Eqs. 1-2); see the
  [API and validation scope](docs/time_dependent.md) and
  [`benchmarks/benchmark_time_dependent.py`](benchmarks/benchmark_time_dependent.py).
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
  more — plus [`notebooks/examples/`](notebooks/examples) (worked analyses such as the BaBar
  2008 `D0 -> KS pi pi` model, the LHCb 2023 `Ds -> 3pi` fit and the BESIII-inspired four-body
  model) and [`notebooks/validation/`](notebooks/validation) (closure and pull studies, and the
  Square-Dalitz reproduction of the LHCb `B -> 3pi` isobar model).

## Installation

```bash
git clone https://github.com/JuanBSLeite/Jax-PWA.git
cd Jax-PWA
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

For NVIDIA GPU support, install exactly one CUDA extra instead of the CPU-only
JAX dependency. The extras use JAX's official CUDA wheels:

```bash
# CUDA 12
python -m pip install -e ".[dev,cuda12]"

# or CUDA 13
python -m pip install -e ".[dev,cuda13]"
```

Do not install both CUDA extras in the same virtual environment. CUDA wheels
are currently provided for Linux; see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for driver and platform requirements.

### Google Colab

Select a GPU runtime (Runtime → Change runtime type), then install straight from
GitHub in the first cell:

```python
!pip install -q "jax-pwa[cuda12] @ git+https://github.com/JuanBSLeite/Jax-PWA.git"
```

Restart the session afterwards (Runtime → Restart session): Colab ships its own JAX,
and the extra may replace it, so the running kernel keeps the old version until it is
restarted. Then check that JAX sees the GPU:

```python
import jax
print(jax.devices())  # expected: [CudaDevice(id=0)]
```

For CPU-only use, drop the extra: `!pip install -q "git+https://github.com/JuanBSLeite/Jax-PWA.git"`.
For a private repository, put a GitHub token in Colab Secrets and use
`git+https://{token}@github.com/...`. Note that `float64` (the project default, see
below) is much slower on the free-tier T4 than on L4/A100 GPUs.

The project deliberately runs `float64`/`complex128` rather than JAX's own default, for
amplitude-analysis stability. `import jaxpwa` already enables double precision
automatically, so no separate call is needed for normal use:

```python
from jaxpwa import enable_x64  # only needed to opt back out

enable_x64(False)  # explicit, unvalidated float32 experiment
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

BaBar Collaboration, *Improved measurement of the CKM angle γ in B<sup>∓</sup> → D<sup>(*)</sup>K<sup>(*)∓</sup>
decays with a Dalitz plot analysis of D decays to K<sub>S</sub><sup>0</sup>π<sup>+</sup>π<sup>-</sup> and
K<sub>S</sub><sup>0</sup>K<sup>+</sup>K<sup>-</sup>*, Phys. Rev. D 78 (2008) 034023, arXiv:0804.2089.

Belle Collaboration, *Measurement of D<sup>0</sup>-D̄<sup>0</sup> mixing and search for indirect CP
violation using D<sup>0</sup> → K<sub>S</sub><sup>0</sup>π<sup>+</sup>π<sup>-</sup> decays*,
Phys. Rev. D 89 (2014) 091103, arXiv:1404.2412.

BESIII Collaboration, *Amplitude analysis of the decays D<sup>0</sup> → π<sup>+</sup>π<sup>-</sup>π<sup>+</sup>π<sup>-</sup>
and D<sup>0</sup> → π<sup>+</sup>π<sup>-</sup>π<sup>0</sup>π<sup>0</sup>*, arXiv:2312.02524.
