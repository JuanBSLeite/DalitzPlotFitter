# Learn DalitzPlotFitter

A progressive course in English, based on the [public API catalog](../docs/catalog.md).
Each notebook contains explanations, executable examples, diagnostic checks and exercises.
Start with lessons 1–2; proceed to normalization before adapting a fit to your own data.
Every lesson is self-contained and generates its own data. The models are illustrative,
not reproductions of published measurements.

## Setup

From the repository root, create or activate a Python environment (Python 3.12 or newer), then:

```bash
python -m pip install -e ".[dev]"
python -m pip install jupyterlab
python -m jupyter lab notebooks
```

Select the kernel belonging to this environment. Run each notebook from top to bottom
in a fresh kernel. Installation is done once in a terminal, not inside the lessons.
`enable_x64()` is called before numerical work. A CPU is sufficient; first-use JAX
compilation can take longer than subsequent evaluations. Runtime depends on the machine.
Examples use modest event counts and grids; increase resolutions and establish convergence
before drawing physics conclusions. Inverse-CDF and normalization resolutions control
separate approximations. No remote data, GPU, or ROOT installation is required.

## Learning path

| Lesson | Notebook | Main public APIs |
|---|---|---|
| 1 | [From a decay channel to a Dalitz plot](tutorial_01_phase_space_and_models.ipynb) | DecayChannel, PhaseSpaceSample, DecayModel, RealImag, plotting |
| 2 | [Your first unbinned amplitude fit](tutorial_02_first_fit.ipynb) | FitSession, Parameter, generate_toy, fit fractions |
| 3 | [Normalization, integration weights and convergence](tutorial_03_normalization.ipynb) | normalization_sample, MassWindowVeto, integration weights |
| 4 | [Efficiency, vetoes, backgrounds and constraints](tutorial_04_acceptance_and_backgrounds.ipynb) | BackgroundSpec, ToyBackground, GaussianConstraint |
| 5 | [Floating dynamics and the low-level fit API](tutorial_05_dynamics_and_low_level_api.ipynb) | PreparedAmplitudeCache, Minimizer, MultiBackgroundNLL |
| 6 | [A joint fit to both charges](tutorial_06_joint_cp_fit.ipynb) | CPRealImag, generate_cp_toy, CPFitSession |
| 7 | [From ROOT events to a reproducible fit](tutorial_07_root_io.ipynb) | write_phase_space_sample, read_phase_space_sample, FitSession.from_root |
| 8 | [Como usar o QMI](tutorial_08_qmi.ipynb) | QMI polar/cartesiano, interpolação, uso em um modelo de Dalitz |
| 9 | [QMI S-wave isobar closure](tutorial_08_qmi_isobar_closure.ipynb) | QMI, magnitude/phase recovery, closure diagnostics |
| 10 | [QMI Cartesian isobar closure](tutorial_09_qmi_cartesian_isobar_closure.ipynb) | Cartesian QMI nodes, coefficient recovery, fit validation |
| 11 | [QMI2D Dalitz-field closure](tutorial_10_qmi2d_dalitz_closure.ipynb) | QMI2D, physical-bin masks, phase maps, two-dimensional closure |

## Continue with focused examples

The course covers the main fitting workflow, not every specialized catalog entry.
Use these existing notebooks and documents for extensions:

| Topic | Next example or reference |
|---|---|
| Multiple backgrounds and extended fits | [Multiple backgrounds](08_b2kpipi_multiple_backgrounds.ipynb), [background conventions](../docs/backgrounds_and_vetoes.md) |
| Detector migration / self-cross-feed | [SCF migration](07_b2kpipi_scf_migration.ipynb), [SCF with veto](12_b2kpipi_scf_with_veto.ipynb) |
| Discriminating variables | [Dalitz plus discriminants](10_b2kpipi_discriminating_variables.ipynb) |
| Square-Dalitz histogram maps | [Efficiency and background maps](15_b2kpipi_square_dalitz_eff_background.ipynb) |
| Repeated toy generation | [Toy generation](18_user_friendly_toy_generation.ipynb), [prepared generators](../docs/toy_generation.md) |
| One-dimensional resolution | [PDF convolution](20_pdf_convolution_resolution.ipynb) |
| Identical-particle folding | [Folded CP example](23_b2pipipi_cp_folded_dalitz_fit.ipynb) |
| Alternative lineshapes, QMI and QMI2D | [Lineshape documentation](../docs/lineshapes.md), [dynamics structure](../docs/dynamics_structure.md) |

The `data_analyses/` and `benchmark/` directories contain analysis work and numerical
reproductions rather than introductory lessons. Consult the
[convention review](../docs/reviews/paper_isobar_conventions.md) for the remaining
publication-reproduction discrepancies in SigmaPole, RhoOmegaMixing and PipiKKRescattering.
