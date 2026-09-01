# Direct-CP coefficient parameterization

DalitzPlotFitter provides `CPRealImag` for charge-conjugate amplitude fits. It is a Cartesian extension of `RealImag` and keeps the CP-averaged amplitude and direct-CP difference in one shared parameter set.

For charge label `q = +1` or `q = -1`,

```text
c_q = (x + q dx) + i (y + q dy).
```

Therefore

```text
c_plus  = (x + dx) + i (y + dy)
c_minus = (x - dx) + i (y - dy).
```

`x` and `y` are the CP-even Cartesian components. `dx` and `dy` are CP-odd differences. Setting `dx = dy = 0` recovers the CP-conserving case.

All four entries may be ordinary numbers or `Parameter.coefficient(...)` objects. The same `Parameter` instances are shared between the two charge models.

## Joint CP likelihood

A direct-CP amplitude fit must preserve both the Dalitz-shape information and the relative rate of the two charge samples. `CPJointNLL` therefore treats charge as part of the fitted sample space.

For coherent charge amplitudes

```text
A_plus(phi)  = sum_j c_plus_j  F_j(phi)
A_minus(phi) = sum_j c_minus_j F_j(phi)
```

define

```text
I_plus  = integral |A_plus|^2 dPhi
I_minus = integral |A_minus|^2 dPhi.
```

The joint signal PDF is

```text
p(phi, +) = |A_plus(phi)|^2  / (I_plus + I_minus)
p(phi, -) = |A_minus(phi)|^2 / (I_plus + I_minus).
```

For unweighted positive- and negative-charge samples the corresponding NLL is

```text
NLL = - sum_plus  log |A_plus|^2
      - sum_minus log |A_minus|^2
      + (N_plus + N_minus) log(I_plus + I_minus).
```

This differs physically from summing two independently normalized Dalitz likelihoods. Independent normalizations would condition on the observed charge counts and remove sensitivity to the integrated charge asymmetry. With `CPJointNLL`, changes in CP coefficients can affect both local interference patterns and the predicted positive/negative yield fractions

```text
P(+) = I_plus  / (I_plus + I_minus)
P(-) = I_minus / (I_plus + I_minus).
```

Within each charge sample all amplitudes are still added coherently before taking the absolute square. The cached normalization uses the full quadratic form `c^dagger M c`, including all interference terms.

When only coefficient parameters float, the dynamical basis and normalization matrices remain cached; only the charge-dependent coefficient vectors and the two quadratic normalizations change.

At least one complex-amplitude convention must be fixed, as in an ordinary amplitude fit, to remove the overall scale/phase degeneracy.

## Complete BaBar B± -> K± pi∓ pi± benchmark

The CP classes remain available in the library, but the reduced tutorial set is
currently restricted to non-CP workflows.

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

- per-component unit Dalitz normalization;
- Blatt-Weisskopf radius `4 GeV^-1`;
- relativistic Breit-Wigner for ordinary resonances;
- LASS with `a=2.07 GeV^-1`, `r=3.32 GeV^-1`, and a `1.8 GeV` effective-range cutoff;
- `BaBarFlatte` for `f0(980)` with the charged/neutral isospin weights used by BaBar;
- a constant complex amplitude for the separate phase-space nonresonant component;
- the fitted scalar `fX` mass and width, `1.479 GeV` and `0.080 GeV`.

For the particle ordering

```text
(1, 2, 3) = (K±, pi±, pi∓)
```

the notebook consistently plots

```text
s13 = m^2(K± pi∓)
s23 = m^2(pi+ pi-).
```

Toy generation follows the same joint PDF as the fit: first the charge is drawn from `I_plus/(I_plus + I_minus)`, then the Dalitz point is generated from the corresponding charge amplitude. The notebook includes charge-separated Dalitz plots, a local 2D CP-asymmetry map, `s13`/`s23` projections, parameter pulls, Argand comparisons, component `A_CP`, and the truth/fit global charge fractions.
