# BaBar 2008 D0 -> KS pi+ pi- amplitude benchmark

Reference: [BaBar, Phys. Rev. D 78, 034023 (2008), arXiv:0804.2089v2](https://arxiv.org/pdf/0804.2089),
Sec. III B-C, Eqs. (7)-(15), Tables I-II. Notebook:
[`babar_2008_d0_kspipi.ipynb`](../../notebooks/benchmark/babar_2008_d0_kspipi.ipynb).

Convention sources used where the paper alone is not sufficient:

- Y. P. Lau, Ph.D. thesis, Princeton University (2007), SLAC-R-872, ref. [42] of the paper
  ([UNT/OSTI 922231](https://digital.library.unt.edu/ark:/67531/metadc901707/)).
  The relevant parts are Eq. (3.21) for the invariant spin factors, footnote 5 for the CLEO A/B typo,
  Table 3.1 and Sec. 4.5.1 for the barriers, Eq. (3.56) for the LASS momentum, and Table 4.5 for masses.
- EvtGen `src/EvtGenModels/EvtD0mixDalitz.cpp`, the complete model (RBW, GS, LASS and K-matrix),
  together with `src/EvtGenBase/EvtDalitzReso.cpp` (<https://gitlab.cern.ch/evtgen/evtgen>, master, read on 2026-09-26).

## Scope and inputs

The notebook constructs the fixed D0 decay amplitude used by the gamma analysis.
It has eight P/D-wave terms, two Kpi S-wave terms and the pipi K-matrix production amplitude.
It does not implement the B-decay likelihood or refit experimental data. All coefficients
are the published Table II central values. No component is rescaled to match a target fit fraction.
The integration measure is ds12 ds13, with `mean(sample.weights * f)` throughout.

The daughter order is `(KS, pi+, pi-)`, so `s12=m_plus^2`, `s13=m_minus^2`, and `s23=m_pipi^2`.
K*(892) and K0*(1430) masses and widths come from the paper. The other masses and widths are PDG 2006
values, as the paper states. K*(1680) uses the LASS K- pi+ n values of 1677 and 205 MeV, not the
PDG average of 1717 and 322 MeV. The numbers are taken from Lau Table 4.5 and agree with EvtGen.

## Findings

**The EvtGen complete model is Table II.** Its coefficients reproduce every Table II magnitude
to the quoted precision. The spin-1 phases agree within 0.5 degrees. Every non-vector phase differs by
180 degrees, within 0.5 degrees. This covers the D waves, the Kpi S wave, beta_1..4 and f11^prod.
This pattern was checked numerically during the audit.

**The spin-1 sign is the corrected CLEO convention.** Note [34] of the paper and footnote 5 of the
thesis say that Kopp et al. (CLEO), Eq. (6), swaps the labels A and B, which flips every spin-1 term.
EvtGen implements the CLEO formula as printed (`qCA - qBC + ...`). Relative to the paper,
that inverts every vector amplitude, including the rho reference. That is exactly the uniform
180-degree shift of the non-vector phases. The paper therefore uses Lau's corrected
Z1 = M_BC^2 - M_AC^2 + (M_D^2 - M_C^2)(M_A^2 - M_B^2)/M_AB^2, with EvtGen's labels:
A = KS for both K*, and A = pi+ for pi+pi- states.

**Spin-factor scale.** Z1 = 4 p q cos(theta_AC) and Z2 = (16/3) p^2 q^2 (3cos^2 - 1) in the resonance frame.
This is `GooFitLegacyAngular` with A as the first index of the pair.
Laura++ `ZemachP` (-2pq cos) is smaller by factors 2 and 4 in amplitude. That changes S/P/D ratios
and cannot be absorbed by the Table II coefficients.

**Kpi S wave.** The printed Eq. (14) puts phi_R inside sin(delta_R). `EvtDalitzReso::lass` puts phi_R only
in the exponential: `R sin(deltaR) exp(i(deltaR+phiR)) exp(2i totalB)`. Only the latter reproduces the
K0*(1430)- fit fraction, 9.8% against 10.2 +/- 1.5%. The literal reading gives 17.7%.
The paper's prose calls q the "spectator" momentum. Lau Eq. (3.56) and EvtGen both use the Kpi breakup
momentum, and so does the notebook.
There is no m/q factor and no mass cutoff, which are EvtGen's defaults.

**Barriers.** F_r is Blatt-Weisskopf normalized at the pole with R = 1.5 GeV^-1.
For the D vertex, the paper's prose mentions a barrier with R = 1.5. The thesis ("We assume F_D = 1")
and EvtGen (default `f_b = 0`) use F_D = 1, which the notebook adopts.

**Gounaris-Sakurai.** EvtGen uses `-i sqrt(s) Gamma(s)` in the denominator. The library's
`GounarisSakurai` uses `-i m0 Gamma(s)`. The notebook defines a local GS with the EvtGen choice.
The effect is below 0.1 percentage points.

**K-matrix.** EvtGen's F-vector matches the library's `KMatrix`. This covers the couplings, f^scatt,
the Adler factor on K only, the production term (1 - s0prod)/(s - s0prod), the 4pi polynomial, and
closed channels continued as i sqrt(-rho^2). Only m_pi and m_eta differ, at the 1e-5 level.

## Applied fixes

- Replace `ZemachP` and raw primed barriers with `GooFitLegacyAngular` and pole-normalized
  resonance barriers. Set F_D = 1.
- Use the corrected CLEO pair orientation. The old notebook mixed `(pi-, KS)` for K*- with
  `(KS, pi+)` for K*+, which is inconsistent under pi+ <-> pi-.
- Revert the earlier "fix" that moved phi_R inside the sine. Use the Kpi breakup momentum in delta_F.
- Replace working-value masses with PDG 2006 and LASS values: K*(1680), rho, omega and f2.
- Use the GS width convention with sqrt(s).

## Numerical validation

The following checks were run during the audit on 2026-09-26. They are not part of the notebook,
which now keeps only the model and the time-dependent workflow of the Belle 2014 notebook.

- **Kpi S wave:** agrees with an independent NumPy transcription at threshold, at the pole and on
  both sides, for both phi_R readings, to 1e-12. The exact threshold uses 1e-8. JAX gradients
  agree with finite differences.
- **K-matrix:** K and P are rebuilt from Tables I-II. The linear solve agrees with `KMatrix`, reusing
  only the library's rho.
- **EvtGen port:** a NumPy port of `EvtD0mixDalitz::dalitzKsPiPi` uses the CLEO printed numerator with
  EvtGen's A/B/C bookkeeping, Blatt-Weisskopf, RBW, GS, LASS and F-vector. It is compared with the
  library model built from EvtGen's coefficients in the paper convention, with non-vector phases
  plus 180 degrees. At 400 random Dalitz points, A_lib = -A_EvtGen/0.97, with a maximum relative
  deviation of 1.7e-13.
- **Normalization:** Hermiticity of the normalization matrix, closure of the coherent integral, and
  projection integrals that close to 4e-4.

Execution on 2026-09-26 used CPU, JAX 0.11.2 and x64. Grids were Square-Dalitz 300 x 300 and 600 x 600.
The coherent integral changed by 0.125%, and the largest FF change was 0.094 pp, which is omega.
Omega moved from 1.00% to 0.90%. It is the one component whose small FF is still resolution sensitive.

Fine-grid fit fractions, with Table II coefficients:

| Component | Local FF (%) | Table II FF (%) | Pull |
|---|---:|---:|---:|
| K*(892)- | 56.16 | 55.7 +/- 2.8 | +0.17 |
| K0*(1430)- | 9.81 | 10.2 +/- 1.5 | -0.26 |
| K2*(1430)- | 2.14 | 2.2 +/- 1.6 | -0.04 |
| K*(1680)- | 0.711 | 0.7 +/- 1.9 | +0.01 |
| K*(892)+ | 0.463 | 0.46 +/- 0.23 | +0.01 |
| K0*(1430)+ | 0.015 | < 0.05 | |
| K2*(1430)+ | 0.009 | < 0.12 | |
| rho(770) | 20.26 | 21.0 +/- 1.6 | -0.46 |
| omega(782) | 0.90 | 0.9 +/- 1.0 | +0.00 |
| f2(1270) | 0.540 | 0.6 +/- 0.7 | -0.09 |
| pipi S | 11.90 | 11.9 +/- 2.6 | +0.00 |
| Sum | 102.9 | 103.6 +/- 5.2 | |

The chi2 over the nine measured FFs is 0.32. It uses published errors, which are dominated by
systematics, and ignores correlations. The notebook asserts |pull| < 1 for all measured FFs and
checks that both DCS limits hold.

One-at-a-time convention study at 300 x 300, run during the audit:

| Variant | Sum FF (%) | chi2 (9 FF) |
|---|---:|---:|
| Nominal | 103.1 | 0.32 |
| phi_R inside the sine, as Eq. (14) is printed | 100.9 | 32.4 |
| CLEO printed sign, spin 1 flipped | 127.9 | 35.5 |
| F_D with R_D = 1.5 | 103.1 | 0.71 |
| GS with -i m0 Gamma(s) | 103.1 | 0.38 |
| `ZemachP` scale | 123.7 | 282.8 |

The uncorrected CLEO sign scales every FF by the same factor, because it only changes the coherent
denominator. It is excluded by the sum of fit fractions, not by FF ratios.

## Time-dependent workflow

The notebook reuses the structure of `belle_2014_d0_kspipi_time_dependent.ipynb` unchanged, with
`TimeDependentFitSession`, the no-direct-CPV reflection and Belle Eqs. (1-2). The injected values are the
BaBar 2010 measurement, x = 1.6e-3 and y = 5.7e-3 (arXiv:1004.5053), with tau = 0.4103 ps.
On 2026-09-26, the Asimov fit returned x = 0.001600 and y = 0.005699. The 20 000-event resampled toy
returned x = (0.30 +/- 0.77)% and y = (0.17 +/- 0.66)%.

## Remaining choices

- F_D = 1 follows the thesis and EvtGen against the paper's prose. The fit fractions mildly prefer it
  but do not exclude R_D = 1.5, which is mostly visible in K*(1680) and f2.
- The GS sqrt(s) width follows EvtGen, while the thesis writes m_R Gamma_R(s). The difference is negligible.
- Table II rounds coefficients to three digits. EvtGen's more precise values come with slightly different
  K*(892) and LASS parameters from the 2010 fit. The nominal FFs use Table II.
- Acceptance, backgrounds and parameter correlations are not available. The integrals are acceptance-free.
