# Direct-CP coefficient parameterization

DalitzPlotFitter provides `CPRealImag` for simultaneous fits of charge-conjugate Dalitz samples. It is a Cartesian extension of `RealImag` and is designed to keep the CP-averaged amplitude and the direct-CP difference in one shared parameter set.

For charge label `q = +1` or `q = -1`,

```text
c_q = (x + q dx) + i (y + q dy).
```

Therefore

```text
c_plus  = (x + dx) + i (y + dy)
c_minus = (x - dx) + i (y - dy).
```

`x` and `y` are the CP-averaged Cartesian coefficient components. `dx` and `dy` are CP-odd differences. Setting `dx = dy = 0` recovers the CP-conserving case.

All four entries may be ordinary numbers or `Parameter.coefficient(...)` objects. The same `Parameter` instances should be shared between the two charge models.

A simultaneous fit is formed by preparing one likelihood/cache for each charge and combining them with `SimultaneousNLL`. When only coefficient parameters float, the dynamical basis and normalization matrices remain cached; only the charge-dependent coefficient vectors and `c^dagger M c` normalization change.

At least one complex-amplitude convention must be fixed, as in an ordinary amplitude fit, to remove the overall scale/phase degeneracy.

## Complete BaBar B± -> K± pi∓ pi± benchmark

`notebooks/12_cp_coefficients_closure.ipynb` implements the complete nominal signal isobar model of BaBar, Phys. Rev. D 78, 012004 (2008), arXiv:0803.4451.

The paper uses exactly the Cartesian convention implemented by `CPRealImag`:

```text
c_j     = (x_j + dx_j) + i (y_j + dy_j)
cbar_j  = (x_j - dx_j) + i (y_j - dy_j)
```

The nominal model contains a constant phase-space nonresonant term and nine intermediate states:

- `K*(892)0 pi`;
- the LASS `(K pi)_0*0 pi` S-wave;
- `K2*(1430)0 pi`;
- `rho(770)0 K`;
- `omega(782) K`;
- `f0(980) K`;
- `f2(1270) K`;
- the scalar `fX(1300) K` term used by BaBar;
- `chi_c0 K`.

The central `x`, `y`, `dx`, and `dy` values are taken directly from Table I of the paper. `K*(892)0` fixes the CP-even reference to `x=1`, `y=0`; its CP-odd shifts remain floating. BaBar fixes `dx=dy=0` for the `omega(782) K` and phase-space nonresonant terms, and the benchmark preserves that choice.

The dynamical conventions reproduced in the fitter are:

- per-component unit Dalitz normalization, matching Eq. (4);
- Blatt-Weisskopf radius `4 GeV^-1`;
- relativistic Breit-Wigner for ordinary resonances;
- LASS with `a=2.07 GeV^-1`, `r=3.32 GeV^-1`, and a `1.8 GeV` effective-range cutoff;
- `BaBarFlatte` for `f0(980)`, with `g_pi=0.165 GeV`, `g_K=4.21*g_pi`, and the charged/neutral isospin weights of Eqs. (10-13);
- a constant complex amplitude for the separate phase-space nonresonant component;
- the fitted scalar `fX` mass and width, `1.479 GeV` and `0.080 GeV`.

For the particle ordering

```text
(1, 2, 3) = (K±, pi±, pi∓)
```

the notebook consistently plots the Dalitz plane in

```text
s13 = m^2(K± pi∓)
s23 = m^2(pi+ pi-)
```

and uses those same coordinates for one-dimensional projections and raw toy asymmetries.

The notebook generates independent positive- and negative-charge pseudo-data samples, prepares one amplitude cache per charge, randomizes the free Cartesian parameters once, performs a single simultaneous Minuit fit, and compares truth/start/fit/pulls together with the reconstructed component CP asymmetries.
