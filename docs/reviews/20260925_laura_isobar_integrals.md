# Direct comparison of isobar integrals with integ_pos.dat / integ_neg.dat

## Result

The nine **isobaric** amplitudes reproduce the supplied Laura++ integral files.
With the logged Laura mass-plane grid and an analytic conversion of barrier
conventions, the largest relative discrepancy of a diagonal integral is
**1.99e-8**. Off-diagonal complex integrals agree to **1.67e-7 relative** or
better. Both bare and efficiency/veto-weighted matrices were compared, for
both charges. No QMI component, fit coefficients, or fitted yields enter this
calculation.

Our native Square-Dalitz grid does not give exactly these numbers: at 2000²
nodes, the largest diagonal differences are 0.0197% bare and 0.3596% accepted.
Reproducing the logged quadrature removes these differences. This validates
the amplitude functions and acceptance convention for the tested isobaric
initialization; it does not establish that the Laura grid is the exact integral.

## What the files contain

Inputs in `/home/juan-leite/Work/data/fit415489_pipipinotav3_21-5--11h32m13s415489/`:

- `integ_pos.dat`, `integ_neg.dat`;
- `GenFit3piCP_R2_415489.cc` and its adjacent fit log;
- `inputs/ACC_pipipi_15161718_NoSpike_prodAsy.root`.

SHA-256 hashes and all numerical matrices are in
`20260925_laura_isobar_integrals_results.json`.

The first four lines identify daughters, nine components, lineshape IDs and
resonance pair IDs. Line 5 contains `integral |F_i|²`; line 6 contains
`integral epsilon*V*|F_i|²`. Lines 7 and 8 contain the upper triangles of
`integral epsilon*V*F_i*conj(F_j)` and `integral F_i*conj(F_j)` respectively.
The extra diagnostic text after these blocks is not needed by our parser.
Each parsed matrix diagonal is checked against its separately printed vector.

`LauIsobarDynamics::initialise` calls `writeIntegralsFile` before the fit.
The numbers also match the initialization summary in the supplied log.
Consequently we compare the **initial catalog dynamics**, not final fitted
masses/widths from the analysis notebook:

| Component | Mass [GeV] | Width [GeV] | Lineshape |
|---|---:|---:|---|
| rho0(770) | 0.77526 | 0.1478 | GS |
| omega(782) | 0.78265 | 0.00849 | RelBW |
| rho0(1450) | 1.465 | 0.400 | GS |
| f_2(1270) | 1.2755 | 0.1867 | RelBW |
| rho0_3(1690) | 1.686 | 0.186 | RelBW |
| sigma0 | 0.475 | 0.550 | SigmaPole |
| f_0(980) | 0.990 | 0.070 | RelBW |
| chi_c0 | 3.41471 | 0.0131 | RelBW, explicit .cc override |
| f_0(1370) | 1.370 | 0.350 | RelBW |

Parent mass 5.27934 GeV and pion mass 0.1395704 GeV come from the local ROOT
catalog (different from `particle`'s current B mass). This candidate catalog
reproduces the supplied integrals to the precision reported here. Radii are
4 GeV^-1, angular model Zemach_P, bachelor momentum in the resonance frame,
pair `(2,0)` in Python and pair ID 2 in Laura. Amplitudes are symmetrized.
The acceptance maps use **bilinear**, folded lookup, with the D0 veto on
both opposite-charge pairs, as specified by the .cc.

## Necessary basis conversion

Our raw resonance amplitude uses pole-normalized barriers:
`sqrt(P_L(z0)/P_L(z))`. Laura's raw amplitude uses `1/sqrt(P_L(z))`.
Thus `F_ours = k_i * F_Laura`, where

`k_i = sqrt(P_L((4*q0)^2) * P_L((4*p0)^2))`.

Here `P_0=1`, `P_1=1+z`, `P_2=9+3z+z²`, and
`P_3=225+45z+6z²+z³`. These factors are derived from the pole momenta,
**not estimated from the .dat integrals**:

| Component | k_i |
|---|---:|
| rho0(770) | 123.6358128 |
| omega(782) | 123.3249725 |
| rho0(1450) | 106.8953707 |
| f_2(1270) | 13744.67162 |
| rho0_3(1690) | 1387859.832 |
| All four scalars | 1 |

For a direct comparison, divide our diagonal by `k_i²` and the interference
entry by `k_i*k_j`. This is a change of amplitude basis, not unit-integral
normalization. Native raw values can be recovered by multiplying the reported
matrices by those factors. The comparison explicitly uses Laura's conjugation
order; our public normalization matrix uses its complex conjugate.

## Bare diagonal integrals on the logged grid

The bare matrices are identical for B+ and B- in the reference files.
The Python column below has the analytic basis conversion applied.

| Component | Laura .dat | Python, Laura basis |
|---|---:|---:|
| rho0(770) | 5.753849520 | 5.753849559 |
| omega(782) | 85.18528121 | 85.18528141 |
| rho0(1450) | 1.539822468 | 1.539822465 |
| f_2(1270) | 0.008378829903 | 0.008378829827 |
| rho0_3(1690) | 0.00002032205883 | 0.00002032205846 |
| sigma0 | 138.3012422 | 138.3012398 |
| f_0(980) | 2300.761760 | 2300.761750 |
| chi_c0 | 2268.402940 | 2268.402935 |
| f_0(1370) | 322.0194412 | 322.0194399 |

Across all entries, the maximum
`abs(M_python-M_Laura)/sqrt(M_Laura_ii*M_Laura_jj)` is 1.84e-8 for the bare
matrix, 1.99e-8 for accepted B+, and 1.98e-8 for accepted B-.
Relative errors measured against each off-diagonal complex entry itself
reach 1.67e-7, 1.19e-7 and 1.12e-7 respectively.

## Our native square grid versus the files

At 2000² nodes, percentages below are `100*(Python/Laura-1)`:

| Component | Bare | Accepted B+ | Accepted B- |
|---|---:|---:|---:|
| rho0(770) | +0.01537 | -0.01441 | -0.01527 |
| omega(782) | +0.00119 | -0.02909 | -0.03154 |
| rho0(1450) | -0.00365 | +0.02056 | +0.02211 |
| f_2(1270) | +0.00615 | +0.01018 | +0.01073 |
| rho0_3(1690) | -0.01973 | +0.35628 | +0.35963 |
| sigma0 | +0.00648 | -0.01246 | -0.01245 |
| f_0(980) | +0.00219 | -0.01094 | -0.01051 |
| chi_c0 | +0.00252 | -0.01994 | -0.01822 |
| f_0(1370) | +0.00029 | +0.00997 | +0.01033 |

The logged grid reproduction uses its 25 mass-plane rectangles, printed
node counts, exact narrow-window endpoints and mass-derived physical endpoints.
It also deliberately reproduces the local Laura Gauss algorithm's stale `zSq`
and 1e-6 Newton stopping criterion. It is a diagnostic path only; the library's
NumPy Gauss rule was not changed. The agreement with the actual .dat files is
the numerical check of this reconstruction, rather than an assumption that
the external executable was identical to the local library.

## Reproduction

The diagnostic now selects `normalize_form_factors=False` directly on each
`Resonance`, rather than dividing the evaluated amplitudes manually. The
analytic factors above remain in the JSON as a reference conversion from the
default convention. Repeating the logged-grid check with the native option
retains the agreement reported above.

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/compare_laura_integral_files.py \
  --output docs/reviews/20260925_laura_isobar_integrals_results.json
```

Runs our actual JAX component functions with coefficients one and component
normalization disabled, at 500²/1000²/2000² and on the logged Laura grid.
The result includes full real/imaginary matrices, per-component diagonal
differences, and entrywise scale-normalized matrix discrepancies.
The script passes Ruff and validates the parsed diagonal vectors against the
matrices. No fit or numerical-library change is part of this comparison.
