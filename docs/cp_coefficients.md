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

## Efficiency and background mixtures

`CPJointNLL` also supports efficiency-weighted signal and a background mixture while preserving the same joint charge normalization.

With charge-dependent efficiencies `epsilon_q(phi)` define

```text
I_plus^eps  = integral epsilon_plus(phi)  |A_plus(phi)|^2  dPhi
I_minus^eps = integral epsilon_minus(phi) |A_minus(phi)|^2 dPhi.
```

The `PreparedAmplitudeCache` objects passed to `CPJointNLL` must therefore be prepared with the corresponding efficiency values on their normalization samples. Event-by-event efficiency arrays are passed separately to `CPJointNLL`.

For background functions `B_plus(phi)` and `B_minus(phi)`, define

```text
J_plus  = integral B_plus(phi)  dPhi
J_minus = integral B_minus(phi) dPhi.
```

With a global background fraction `f_bkg`, the full joint PDF is

```text
p(phi,+) = (1-f_bkg) epsilon_plus(phi)|A_plus(phi)|^2
                         / (I_plus^eps + I_minus^eps)
           + f_bkg B_plus(phi)/(J_plus + J_minus)

p(phi,-) = (1-f_bkg) epsilon_minus(phi)|A_minus(phi)|^2
                         / (I_plus^eps + I_minus^eps)
           + f_bkg B_minus(phi)/(J_plus + J_minus).
```

Thus both signal and background are normalized in the combined `(Dalitz, charge)` space. The fitted total charge probabilities are

```text
P(+) = (1-f_bkg) I_plus^eps/(I_plus^eps + I_minus^eps)
       + f_bkg J_plus/(J_plus + J_minus)

P(-) = (1-f_bkg) I_minus^eps/(I_plus^eps + I_minus^eps)
       + f_bkg J_minus/(J_plus + J_minus).
```

The signal-only API remains unchanged:

```python
nll = CPJointNLL(plus_cache, minus_cache)
```

For the complete mixture, use for example

```python
nll = CPJointNLL(
    plus_cache,
    minus_cache,
    plus_efficiency=eff_plus_data,
    minus_efficiency=eff_minus_data,
    plus_background=bkg_plus_data,
    minus_background=bkg_minus_data,
    plus_background_normalization=bkg_plus_norm,
    minus_background_normalization=bkg_minus_norm,
    background_fraction=background_fraction,
)
```

`background_fraction` may be either a number or a fit `Parameter`.

## B± -> K± pi+ pi- tutorial convention

The CP tutorial notebooks use the particle ordering

```text
(1, 2, 3) = (K±, pi±, pi∓)
```

and consistently display

```text
s13 = m^2(K± pi∓)
s23 = m^2(pi+ pi-).
```

The signal-only tutorial uses `CPJointNLL` directly. The efficiency/background tutorial uses the extended form above, with charge kept as part of the fitted sample space throughout. Both tutorials start the minimization from deliberately displaced values and report generated, start and fitted coefficients together with charge-separated fit fractions.
