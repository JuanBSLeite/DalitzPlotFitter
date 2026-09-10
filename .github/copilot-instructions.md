# Copilot instructions for DalitzPlotFitter

## What this is

DalitzPlotFitter is an unbinned amplitude-fit package for three-body decays. The numerical
pipeline is **JAX end to end**: phase-space generation, kinematics, amplitude dynamics,
normalization, likelihoods and gradients all run as JAX arrays/ops. `iminuit` only performs the
final minimization, consuming a JAX `value_and_grad` objective. `particle` supplies standard
particle properties and `uproot` provides ROOT I/O without PyROOT. There is no TensorFlow
dependency anywhere in the numerical path.

Laura++ is the primary physics reference for resonance, barrier-factor, angular and
Square-Dalitz conventions, but classes use neutral names rather than Laura++-specific ones.
When a convention is ported from Laura++ (e.g. `Rescattering2`, the K-matrix, `Zemach_P`/
`Zemach_Pstar`), the docstring says so explicitly and any deliberate deviation from the
upstream C++ (e.g. a typo it contains) is documented and justified in the docstring — don't
assume the literal upstream formula is followed if the docstring says otherwise.

## Commands

```bash
python -m pip install -e ".[dev]"     # install with test + ruff extras

pytest                                 # full suite (testpaths = tests/, ~390 tests, few minutes)
pytest tests/test_rescattering2.py -v  # single file
pytest tests/test_rescattering2.py::test_rescattering2_matches_laura_reference_points  # single test

ruff check .                           # lint (configured, not CI-enforced; repo has existing lint debt,
                                        # so check new/changed files rather than expecting a clean run)

python benchmarks/benchmark_fit_evaluation.py --events 100000 --normalization-resolution 1000 --repeats 20
python benchmarks/benchmark_toy_generation.py --size 100000
python benchmarks/benchmark_scf_migration.py --bins-mprime 40 --bins-thetaprime 40
python benchmarks/benchmark_cache_stages.py --events 100000 --normalization-resolution 1000
python benchmarks/benchmark_qmi_memory_speed.py
```

CI (`.github/workflows/tests.yml`) runs `pytest tests` on Python 3.10 and 3.12, plus a notebook
sanity check that parses every `notebooks/*.ipynb` with `nbformat` and compiles (not executes)
each code cell. `full-validation.yml` and `toy-benchmark.yml` are `workflow_dispatch`-only and
not run on every push. There is no GPU CI; `docs/gpu_ubuntu_24_04.md` documents the manual
WSL2/CUDA reference environment used for GPU validation.

Always `enable_x64()` before numerical work (`from dalitzplotfitter import enable_x64`) — the
project deliberately runs float64/complex128 for amplitude-analysis stability, and this is not
the JAX default.

## Architecture

### Pipeline

```
PhaseSpaceSample (data or generated)
  -> DecayChannel + amplitude components (Resonance / NonResonant / DalitzAmplitude)
  -> pure-JAX kinematics and dynamics
  -> deterministic Dalitz / Square-Dalitz quadrature, or external toy-MC normalization
  -> coherent amplitude -> PreparedAmplitudeCache
  -> optional efficiency/veto/SCF/backgrounds/discriminating-variable PDFs/1D convolution
  -> optional Gaussian constraints
  -> FitSession / CPFitSession (optional convenience layer)
  -> JAX NLL + automatic gradient -> Minimizer (iminuit)
```

### Two API layers, by design

`FitSession`/`CPFitSession` (`workflow.py`, `cp_workflow.py`) are *composition* layers over the
low-level public classes (`SignalPDF`, `PreparedAmplitudeCache`, `MultiBackgroundNLL`,
`CPJointNLL`, `Minimizer`) — they do not replace them. Advanced/validation work should still go
through the low-level classes directly; see `docs/user_friendly_api.md` "Design principle".

### One-dimensional lineshapes vs. full 2D Dalitz amplitudes

`Resonance` composes a lineshape through the ordinary `lineshape(mass, context)` interface
(`dynamics/lineshape/*.py`, one physics model per file: `relativistic_breit_wigner.py`,
`gounaris_sakurai.py`, `flatte.py`, `pole.py`, `lass.py`, `kmatrix.py`, `qmi.py`,
`rescattering2.py`) combined with an angular factor (`dynamics/angular.py`) and Blatt-Weisskopf
barriers. `DalitzAmplitude` bypasses that isobar construction entirely for amplitudes that are
intrinsically two-dimensional (`QMI2D`, `dynamics/qmi2d.py`), evaluated directly over
`(s12, s13)`.

### Normalization: the central invariant

Every integral in the codebase (grid quadrature *and* Monte Carlo) follows one convention:
`integral(f) = mean(sample.weights * f)`. This is why `PhaseSpaceSample` weight semantics matter
so much — an unweighted signal toy is *not* generally a valid unit-weight integration sample; see
`docs/mc_integration.md` before touching anything that consumes `normalization_sample`.

Three `DecayModel` normalization methods exist (`normalization_method=`): `gauss-legendre`
(default, mass-plane quadrature), `square-dalitz` (Square-Dalitz quadrature, matches Laura++'s
prescription for diagonal narrow structures), and `toy-mc` (auto-selected when
`normalization_sample=` is supplied). Narrow resonances (width <= 20 MeV by default) trigger
automatic local grid refinement; see the "Normalization" section of `README.md`.

Two independent normalization switches exist and are easy to conflate:
- `DecayModel(normalize_components=...)` — model-wide default for per-component unit-integral
  rescaling before the complex coefficient is applied.
- `Resonance(..., normalize_component=True|False|None)` — per-component override; `None`
  inherits the model default. A component with `normalize_component=False` still enters the full
  Hermitian normalization matrix `M_ij = integral conj(F_i) F_j dPhi` (interference is never
  skipped), only its own unit-integral rescaling is skipped.

### Prepared caching is a performance-critical pattern, not an implementation detail

`PreparedAmplitudeCache` (`amplitude/cache.py`) separates what is fixed for a Minuit step
(component values + normalization matrix on the data/normalization sample) from what must be
recomputed when a `ParameterKind.DYNAMICS` parameter floats (only the affected component and its
normalization-matrix row/column). Coefficient-only fits never re-evaluate lineshapes or
re-integrate the Dalitz plot after the first prepare. When changing anything in the fit hot path
(`decay.py`, `amplitude/cache.py`, `likelihood/*.py`), check `docs/performance.md` for which
computation is supposed to be cached vs. recomputed, and re-run the relevant benchmark rather
than assuming a change is free.

### CP fits share one normalization across charges

`CPJointNLL` (`likelihood/cp.py`) treats charge as part of the fitted sample space: B+/B- are
normalized jointly (`p(phi,+) = |A_plus|^2 / (I_plus + I_minus)`), not as two independently
normalized Dalitz likelihoods. This is what gives the fit sensitivity to the integrated charge
asymmetry. `cp_workflow.py`'s projection helpers must preserve this — e.g.
`_joint_scaled_weights` divides each charge's projection sample by its *own* size before summing,
since adaptive/external MC samples for B+ and B- need not have equal size. See
`docs/cp_coefficients.md`.

### Toy generation: two independent public samplers

`inverse-transform` (default; numerical Rosenblatt transform, tabulated CDFs, `docs/toy_generation.md`)
and `accept-reject` (Laura++-style envelope/restart algorithm, explicit `method="accept-reject"`).
They are validated against each other, not derived from one another — don't assume one is a
special case of the other.

## Where to look before changing behavior

Each subsystem has one focused doc under `docs/` (`fitting.md`, `lineshapes.md`,
`mc_integration.md`, `backgrounds_and_vetoes.md`, `cp_coefficients.md`, `scf.md`,
`square_dalitz.md`, `toy_generation.md`, `discriminants_and_constraints.md`,
`convolution_resolution.md`, `dynamics_structure.md`, `performance.md`, `root_io.md`,
`user_friendly_api.md`). `docs/reviews/` contains dated, adversarial numeric-reproduction review
write-ups (concrete inputs, reproduced numbers, "Applied fixes" sections) — this repo's working
style is to reproduce a suspected discrepancy numerically before changing formulas, and to update
the corresponding `docs/*.md` in the same change that fixes the code. `notebooks/` (root) are the
tutorial/example set referenced by the docs and README; `notebooks/data_analyses/` holds
in-progress physics analyses (not tutorials) that consume the same public API and can break
silently when a lineshape or normalization convention changes underneath them.
