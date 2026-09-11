# API catalog

An index of every name exported from `dalitzplotfitter` (i.e. `import dalitzplotfitter as m;
m.__all__`): what it is, one line on what it does, and where to read more. Everything below is
importable directly as `from dalitzplotfitter import <Name>` — no submodule path needed.

This page is a map, not a tutorial. For conventions, formulas and worked examples, follow the
"docs" links to the per-topic file, and the "notebooks" links to a runnable example. For anything
not covered here (private helpers, submodule-only exports like the `Simultaneous`/`Unbinned`/
`WeightedUnbinned` NLL variants in `dalitzplotfitter.likelihood`), read the module directly —
this catalog only covers the public top-level surface.

For a guided introduction, start with the [seven-part notebook course](../notebooks/TUTORIALS.md).
The lessons explain the main fitting workflow and link to specialized examples below.

## Model construction

| Name | Kind | What it does |
|---|---|---|
| `DecayChannel` | class | Parent particle and ordered three-body final state (`("D+", ("pi-","pi+","pi+"))`). |
| `DecayModel` | class | Build a coherent amplitude model with deterministic Dalitz-plane normalization; owns `normalization_method`/`normalize_components`. |
| `Resonance` | class | Declarative one-dimensional resonance component: mass, width, spin, coefficient, and an interchangeable `lineshape`/`angular` plugin. |
| `NonResonant` | class | Constant (S-wave, isotropic) non-resonant component with a complex coefficient. |
| `DalitzAmplitude` | class | Attach a genuinely two-dimensional amplitude (e.g. `QMI2D`) that depends on both Dalitz invariants at once, bypassing the isobar construction. |
| `AmplitudeComponent` | class | Named dynamical component `F_i(x)` with a coefficient; the base type `Resonance`/`NonResonant`/`DalitzAmplitude` all produce. |
| `CoherentAmplitudeModel` | class | Coherent sum `A(x) = sum_i c_i F_i(x)` of prepared components. |
| `ConstantAmplitude` | class | Non-resonant constant dynamical amplitude (the piece `NonResonant` wraps). |
| `PreparedAmplitudeCache` | class | Pre-evaluated component values and Hermitian normalization matrix for a data/normalization sample; the object that makes repeated NLL evaluation cheap. |

Docs: `docs/fitting.md`, `docs/dynamics_structure.md`, `docs/performance.md` (caching). Notebooks: `01_e791_toy_fit.ipynb`, `03_b2kpipi_toy_fit.ipynb`.

## Complex coefficients

| Name | Kind | What it does |
|---|---|---|
| `RealImag` | class | Complex coefficient `c = x + i y` for one amplitude component. |
| `CPRealImag` | class | Cartesian CP coefficient `c_q = (x + q*dx) + i(y + q*dy)` shared between the B+/B- charge models. |

Docs: `docs/fitting.md` ("RealImag coefficients"), `docs/cp_coefficients.md`.

## One-dimensional resonance dynamics (lineshapes)

Each is passed as `Resonance(..., lineshape=...)`; all implement `lineshape(mass, context)`.

| Name | Kind | What it does |
|---|---|---|
| `RelativisticBreitWigner` | class | Standard relativistic Breit-Wigner with running width and Blatt-Weisskopf barrier factor. |
| `Pole` | class | Simple fixed-width Breit-Wigner pole, `1/(m - m0 - i*Gamma0/2)` (Laura++ `BW`). |
| `SigmaPole` | class | LHCb 3pi-isobar `f0(500)` pole, `1/((pole_mass - i*pole_width)**2 - m**2)`; distinct from `Pole`'s convention. |
| `GounarisSakurai` | class | Gounaris-Sakurai lineshape for rho-like vector states. |
| `RhoOmegaMixing` | class | Coherent rho-omega mixing (LHCb 3pi-isobar Eq. 15); `component="rho"`/`"omega"` selects one of the two effective split terms. |
| `PipiKKRescattering` | class | LHCb 3pi-isobar phenomenological pi-pi to K-Kbar S-wave (Eqs. 17-21), zero outside `1.0-1.5 GeV`; `convention="paper"` (default) or `"laura"`. |
| `Flatte` | class | Coupled two-channel Flatte lineshape (generic; construct channel masses directly). |
| `BaBarFlatte` | class | Flatte form as parameterized in the BaBar `B± -> K± pi∓ pi±` analysis (arXiv:0803.4451). |
| `LASS` | class | Effective-range + `K0*(1430)` coherent S-wave form for `K pi`. |
| `KMatrix` | class | Five-pole, five-channel Anisovich-Sarantsev pi-pi S-wave K-matrix; exposes `scattering_amplitude()`/`s_matrix()` for unitarity checks. |
| `QMI` | class | Quasi-model-independent S-wave specified at fixed mass knots; `interpolation=` selects `linear`/`cubic`/`hermite`/`natural`, polar or Cartesian knot parameters. |
| `Rescattering2` | class | Port of Laura++ `LauRescattering2Res`: two-region Chebyshev pi-pi/KK rescattering S-wave, zero below the `2*m_K` threshold. |

Docs: `docs/lineshapes.md` (formulas + references). Notebooks: `notebooks/data_analyses/22_rescattering2_toy.ipynb`
(Rescattering2 diagnostics), `notebooks/benchmark/paper_isobar_benchmark.ipynb` and
`notebooks/benchmark/paper_isobar_benchmark_squaredp_01.ipynb` (`SigmaPole`/`RhoOmegaMixing`/
`PipiKKRescattering` reproduction of the LHCb `B -> 3pi` isobar model, Phys. Rev. D 101, 012006).
`docs/reviews/paper_isobar_conventions.md` documents the numeric reproduction, the confirmed
angular-orientation/ACP fixes, and the remaining unresolved discrepancies in that benchmark.

## Angular models

Passed as `Resonance(..., angular=...)`; default is `CovariantAngular()`.

| Name | Kind | What it does |
|---|---|---|
| `CovariantAngular` | class | Default covariant angular factor (spin `L=0..4`). |
| `ZemachP` / `Zemach_P` | class | Laura++ `Zemach_P`: bachelor momentum `p` evaluated in the resonance rest frame. Both names are the same class. |
| `ZemachPstar` / `Zemach_Pstar` | class | Laura++ `Zemach_Pstar`: bachelor momentum `p*` evaluated in the parent rest frame. Both names are the same class. |
| `GooFitLegacyAngular` | class | Legacy GooFit `Ds -> pi pi pi` angular convention (`L=0..2` only), for reproducing historical fits. |

Docs: `docs/dynamics_structure.md`.

## Resonance assembly internals

| Name | Kind | What it does |
|---|---|---|
| `ResonanceAmplitude` | class | Complete one-dimensional resonance amplitude assembled from a lineshape + angular + barrier plugin; what `Resonance` builds under the hood. |
| `ResonanceContext` | class | Kinematic/particle-property bundle (`parent_mass`, `daughter_masses`, `bachelor_mass`, `spin`, `pole_mass`, `pole_width`, radii) passed to lineshape/angular callables. |

Docs: `docs/dynamics_structure.md`, `docs/lineshapes.md`.

## Two-dimensional Dalitz amplitudes

| Name | Kind | What it does |
|---|---|---|
| `QMI2D` | class | Complex amplitude field defined bin-by-bin over `(s12, s13)`; three interpolation modes (`none`/`linear`/`cubic`), optional identical-particle folding. |
| `physical_bin_mask` | function | Mark which `QMI2D` grid cells intersect the physical Dalitz boundary, using the exact analytic boundary. |

Docs: `docs/lineshapes.md` ("QMI2D Dalitz amplitude").

## Kinematics and phase space

| Name | Kind | What it does |
|---|---|---|
| `PhaseSpaceSample` | class | Core data container: invariants, weights, optional four-momenta. Every integral in the package is `mean(sample.weights * f)`. |
| `PhaseSpaceMC` | class | Generate weighted three-body phase-space events in the physical weight convention `ds12 ds13 / (128*pi^3*M^2)`. |
| `CovariantKinematics` | class | Bundle of `resonance_mass, p_star, p, q, cos_theta` used by covariant/Zemach angular factors. |
| `covariant_kinematics` | function | Compute covariant-spin kinematics from daughter four-vectors. |
| `covariant_kinematics_from_invariants` | function | Compute the same kinematics directly from Dalitz invariants (no four-vectors needed). |
| `boost_to_rest_frame` | function | Lorentz-boost a four-momentum into the rest frame of another. |
| `dalitz_s13_limits` | function | Exact physical `s13` bounds at fixed `s12` (the Dalitz boundary). |

Docs: `docs/mc_integration.md`, `docs/toy_generation.md`.

## Square-Dalitz coordinates and quadrature

| Name | Kind | What it does |
|---|---|---|
| `SquareDalitzGrid` | class | Laura++-convention `(m', theta')` deterministic integration grid with Jacobian weights folded into `PhaseSpaceSample.weights`. |
| `DalitzGaussLegendreGrid` | class | Tensor-product Gauss-Legendre grid directly in `(m13, m23)` (the default `normalization_method="gauss-legendre"` engine). |
| `invariants_to_square_dalitz` | function | Convert `(s12,s13,s23)` to Laura++ `(m', theta')`. |
| `square_dalitz_to_invariants` | function | Inverse of the above. |
| `square_dalitz_jacobian` | function | Absolute Jacobian `|d(s_ij,s_ik)/d(m',theta')|` for the SDP map. |
| `fold_thetaprime` | function | Map `theta'` onto `[0, 0.5]` (identical-particle exchange fold); used by `SquareDalitzHistogramEfficiency`/`Background(folded=True)` and `plot_square_dalitz(folded=True)`, see `docs/backgrounds_and_vetoes.md`. |

Docs: `docs/square_dalitz.md`.

## Fit parameters and minimization

| Name | Kind | What it does |
|---|---|---|
| `Parameter` | class | Configuration for one scalar fit parameter (value, bounds, `owner`, `ParameterKind`); use `Parameter.coefficient(...)`/`Parameter.dynamics(...)` constructors. |
| `ParameterKind` | class | Enum-like role tag (`COEFFICIENT` vs `DYNAMICS`) that determines what a floating parameter invalidates in the cache. |
| `Minimizer` | class | Wraps `iminuit` around a JAX `value_and_grad` objective; `fit()`, `fit_multistart()`, `check_gradient()`. |
| `MultiStartResult` | class | Collection of independent minimizations from `fit_multistart`, plus the best valid minimum. |

Docs: `docs/fitting.md`.

## Low-level PDFs and likelihoods

These are what `FitSession`/`CPFitSession` compose automatically; use them directly for custom or CP workflows the convenience layer doesn't cover.

| Name | Kind | What it does |
|---|---|---|
| `SignalPDF` | class | Efficiency-corrected, normalized signal density built from a `PreparedAmplitudeCache`. |
| `SCFSignalPDF` | class | Signal PDF including correctly-reconstructed *and* self-cross-feed (SCF) migrated events. |
| `MultiBackgroundNLL` | class | Unbinned NLL: signal plus an arbitrary number of named background categories (non-CP). |
| `CPJointNLL` | class | Unbinned NLL for simultaneous B+/B- fits with one joint `(Dalitz, charge)` normalization — charge is part of the sample space, not fit independently per charge. |

Docs: `docs/fitting.md`, `docs/cp_coefficients.md`, `docs/backgrounds_and_vetoes.md`, `docs/scf.md`, `docs/performance.md`.

## Backgrounds

| Name | Kind | What it does |
|---|---|---|
| `BackgroundCategory` | class | One normalized background category (density + normalization integral) for `MultiBackgroundNLL`. |
| `BackgroundSpec` | class | Background *shape* for `FitSession`, normalized automatically on the fit's own measure — no manual integral needed. |
| `CPBackgroundCategory` | class | One background category in the joint `(Dalitz, charge)` space, for `CPJointNLL`. |
| `CPBackgroundSpec` | class | Charge-aware background shape for `CPFitSession` (`plus_shape`/`minus_shape`, or one shared shape). |
| `ToyBackground` | class | One background component for high-level (`generate_toy`) non-CP toy generation. |
| `CPToyBackground` | class | Charge-aware background component for `generate_cp_toy`. |

Docs: `docs/backgrounds_and_vetoes.md`, `docs/user_friendly_api.md`, `docs/toy_generation.md`.

## Vetoes

| Name | Kind | What it does |
|---|---|---|
| `VetoMap` | class | Base type for a binary Dalitz-acceptance mask; `.apply(sample)` selects data, `.apply(sample, for_integration=True)` selects and rescales an integration sample. |
| `MassWindowVeto` | class | Reject one invariant-mass window, Laura++ `addMassVeto` convention (bounds in GeV, not GeV²). |
| `CompositeVeto` | class | Logical AND of any number of veto maps. |
| `FunctionalVeto` | class | Wrap an arbitrary callable returning accept/reject per event as a veto map. |
| `VetoedDensity` | class | Apply a veto map to any density/shape callable (e.g. a background shape). |
| `vetoed_signal_pdf` | function | Build a veto-aware `SignalPDF` directly from a `DecayModel`. |

Docs: `docs/backgrounds_and_vetoes.md`.

## Self-cross-feed (SCF)

| Name | Kind | What it does |
|---|---|---|
| `SquareDalitzSCFMap` | class | Uniform Square-Dalitz SCF fraction + true->reconstructed migration map (dense or sparse). |
| `SparseMigration` | class | COO representation of `P(reco_bin \| true_bin)` for large migration maps; `O(nnz)` memory instead of `O(n_bins^2)`. |

Docs: `docs/scf.md`.

## Efficiency/background maps from ROOT histograms

| Name | Kind | What it does |
|---|---|---|
| `SquareDalitzHistogramEfficiency` | class | Piecewise-constant efficiency map in `(m', theta')`, constructed directly from arrays (no ROOT needed). |
| `SquareDalitzHistogramBackground` | class | Piecewise-constant background shape in `(m', theta')`, constructed directly from arrays. |
| `square_dalitz_efficiency_from_root` | function | Build a `SquareDalitzHistogramEfficiency` from a ROOT TH2 with `(m', theta')` axes. |
| `square_dalitz_background_from_root` | function | Build a `SquareDalitzHistogramBackground` from a ROOT TH2 with `(m', theta')` axes. |
| `histogram_efficiency_from_root` | function | Build an efficiency map from a ROOT TH2 in ordinary `(s_ij, s_ik)`-style Dalitz coordinates. |
| `histogram_background_from_root` | function | Build a background map from a ROOT TH2 in ordinary Dalitz coordinates. |
| `read_root_histogram2d` | function | Low-level: read a ROOT TH2 into `(values, x_edges, y_edges)` JAX arrays via uproot. |

Docs: `docs/root_io.md`. Notebooks: `15_b2kpipi_square_dalitz_eff_background.ipynb`.

## ROOT tree I/O

| Name | Kind | What it does |
|---|---|---|
| `read_root_tree` | function | Read arbitrary named ROOT TTree branches into JAX arrays, with an optional `cut`. |
| `read_phase_space_sample` | function | Read a ROOT TTree directly into a `PhaseSpaceSample` (`s12/s13/s23`, optional `weight` and four-momenta). |
| `write_phase_space_sample` | function | Write one `PhaseSpaceSample` to a ROOT TTree with uproot. |
| `write_phase_space_samples` | function | Write several `PhaseSpaceSample` objects to ROOT TTrees in one call. |
| `write_cp_phase_space_sample` | function | Write B+ and B- samples to one TTree with a signed `charge` branch. |

Docs: `docs/root_io.md`. Notebooks: `13_b2kpipi_root_tree_input.ipynb`, `19_toy_root_output.ipynb`.

## Toy (pseudo-data) generation

| Name | Kind | What it does |
|---|---|---|
| `generate_toy` | function | Generate signal/background pseudo-data; `method="inverse-transform"` (default) or `"accept-reject"`. |
| `generate_signal_toy` | function | Generate an unweighted signal-only toy. |
| `generate_cp_toy` | function | Generate a CP toy for both charges at once, with the accepted-integral charge split and optional single-ROOT-file output. |
| `prepare_inverse_toy_generator` | function | Precompute the inverse-CDF tables once for repeated toys at fixed model parameters. |
| `PreparedInverseToyGenerator` | class | The reusable object `prepare_inverse_toy_generator` returns; `.generate(n, seed=...)`. |
| `weighted_resample` | function | Draw unweighted events from a weighted phase-space sample (the resampling building block behind `method="resample"`). |

Docs: `docs/toy_generation.md`. Notebooks: `18_user_friendly_toy_generation.ipynb`, `19_toy_root_output.ipynb`.

## Discriminating-variable PDFs and external constraints

| Name | Kind | What it does |
|---|---|---|
| `Gaussian1D` | class | Gaussian PDF normalized on a finite interval. |
| `Exponential1D` | class | Exponential PDF `exp(slope*x)` normalized on a finite interval. |
| `Histogram1D` | class | Piecewise-constant normalized histogram PDF from edges + values. |
| `BreitWigner1D` | class | Constant-width Breit-Wigner PDF normalized on a finite mass interval. |
| `LineshapeIntensity1D` | class | Turn an existing complex dynamics lineshape (e.g. `RelativisticBreitWigner`) into a normalized 1D intensity PDF. |
| `FactorizedDensity` | class | Multiply a base Dalitz density by independent 1D discriminant PDFs (mass, BDT, PID, ...). |
| `GaussianConstraint` | class | Gaussian penalty `0.5*((x-mu)/sigma)^2` on one parameter. |
| `ConstrainedNLL` | class | Add one or more `GaussianConstraint`s to an existing NLL. |

Docs: `docs/discriminants_and_constraints.md`. Notebooks: `10_b2kpipi_discriminating_variables.ipynb`, `11_b2kpipi_gaussian_constraints.ipynb`.

## Detector-resolution convolution (1D)

| Name | Kind | What it does |
|---|---|---|
| `ConvolvedPDF1D` | class | Numerically convolve a normalized 1D PDF with a resolution kernel, with finite-observed-window normalization. |
| `GaussianResolution1D` | class | Gaussian conditional resolution kernel `R(x_obs \| x_true)`. |

Docs: `docs/convolution_resolution.md`. Notebooks: `20_pdf_convolution_resolution.ipynb`.

## Plotting

| Name | Kind | What it does |
|---|---|---|
| `plot_dalitz` | function | Plot a 2D Dalitz histogram in one call. |
| `plot_square_dalitz` | function | Plot a 2D Square-Dalitz histogram from ordinary invariant coordinates. |
| `plot_binned_data` | function | Plot 1D data as black points with statistical error bars. |
| `binned_data` | function | Return bin centers, counts, uncertainties and edges without plotting (for custom figures). |

Docs: `docs/user_friendly_api.md` ("Automatic projections", "Plot helpers").

## High-level sessions

Composition layers over everything above; see `docs/user_friendly_api.md` "Design principle" for what they intentionally do *not* replace.

| Name | Kind | What it does |
|---|---|---|
| `FitSession` | class | Compose PDF + likelihood + backgrounds + constraints + minimizer for one sample in a few lines; `fit()`, `report()`, `plot_projection()`, `.from_root(...)`. |
| `CPFitSession` | class | Same composition for simultaneous B+/B- fits over `CPJointNLL`; shared `Parameter`s collected once. |

Docs: `docs/user_friendly_api.md`. Notebooks: `16_user_friendly_quickstart.ipynb`, `17_b2kpipi_cp_user_friendly.ipynb`.

## Configuration

| Name | Kind | What it does |
|---|---|---|
| `enable_x64` | function | Enable (default) or disable JAX 64-bit floating-point precision. Call this before any numerical work. |

Docs: `README.md` "Installation", `docs/fitting.md`.
