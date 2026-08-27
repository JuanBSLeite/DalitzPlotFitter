# Laura++ resonance dynamics and angular terms

DalitzPlotFitter implements its resonance physics directly in JAX following Laura++ conventions. There is no parallel AmpForm or symbolic dynamics path.

## Model flow

```text
weighted phasespace MC
  -> four-momenta and invariants
  -> Laura++ covariant kinematics
  -> resonance line shape R(m)
  -> running width Gamma(m)
  -> Blatt-Weisskopf factors
  -> Covariant angular factor T_L
  -> automatic identical-particle symmetrization
  -> complete component F_i(x)
  -> RealImag coefficient c_i = x_i + i y_i
  -> coherent amplitude A = sum_i c_i F_i
  -> weighted-MC normalization
```

## Covariant angular formalism

For

```text
P -> R b
R -> d1 d2
```

define `p*` as the bachelor momentum in the parent rest frame, `p` as the bachelor momentum in the resonance rest frame, `q` as the selected resonance-daughter momentum in the resonance rest frame, and `theta` as the angle between that daughter and the bachelor in the resonance rest frame.

`covariant_spin_factor()` implements Laura++ Eqs. (91)-(95) for `L=0..4`:

```text
T0 = 1

T1 = -2 (p* q) sqrt(1 + p^2/mP^2) cos(theta)

T2 = (4/3) (p* q)^2 (3/2 + p^2/mP^2)
     [3 cos^2(theta) - 1]

T3 = -(24/15) (p* q)^3 sqrt(1 + p^2/mP^2)
     (5/2 + p^2/mP^2)
     [5 cos^3(theta) - 3 cos(theta)]

T4 = (16/35) (p* q)^4
     [8 p^4/mP^4 + 40 p^2/mP^2 + 35]
     [35 cos^4(theta) - 30 cos^2(theta) + 3]
```

The signs and numerical factors are part of the amplitude convention. `covariant_kinematics(daughter, partner, bachelor)` computes the required quantities directly from `(E, px, py, pz)` four-vectors. The selected daughter fixes the sign of odd-spin terms.

## Relativistic Breit-Wigner

The current resonance line shape is

```text
R(m) = 1 / (m0^2 - m^2 - i m0 Gamma(m))
```

with

```text
Gamma(m) = Gamma0 (q/q0)^(2L+1) (m0/m) X_L(q r)^2.
```

The breakup momentum is

```text
q(m) = sqrt(lambda(m^2,m1^2,m2^2)) / (2m)
```

and the Blatt-Weisskopf factor is normalized so `X_L(q0 r)=1`.

## Complete component

`LauraCovariantRBW` evaluates

```text
F_i(x) = R_i(m) X_parent X_resonance T_L(x).
```

A resonance is specified once by its nominal `daughter_key`, `partner_key` and `bachelor_key`. When `final_state` is supplied, the component checks for repeated particle labels and coherently sums all equivalent assignments generated only by exchanges of identical final-state particles.

For example, with

```text
final_state = ("pi-", "pi+", "pi+")
nominal pairing = (p1,p2)p3
```

the component evaluates

```text
F = F[(12)3] + F[(13)2].
```

No external Bose-symmetrization wrapper is required. A constant non-resonant term remains a single constant and is therefore not multiplied by the number of identical-particle permutations.

## Monte Carlo normalization

`PhasespaceMC` requests raw weights using `normalize_weights=False`. For a coefficient-only model,

```text
M_ij = (1/N_MC) sum_k w_PS,k F_i*(x_k) F_j(x_k)
N(c) = c^dagger M c.
```

The reference normalization sample size is **1,000,000 weighted events**.

## Pseudo-data

For generated parameters, candidate weights are

```text
w_target,k = w_PS,k |A(x_k)|^2.
```

`weighted_resample()` converts a larger weighted candidate pool into ordinary unweighted pseudo-data. The reference fit sample is **100,000 events**.

## Validation

The numerical implementation is tested for:

```text
Gamma(m0) = Gamma0
X(q0 r) = 1
T0 = 1
```

as well as the published Covariant expressions, Lorentz boosts, daughter-exchange parity, phase-space four-momentum conservation, weighted resampling, and equality between automatic identical-particle symmetrization and the corresponding explicit sum of pairings.

The next end-to-end milestone is the 100k/1M `RealImag` closure test on this Laura++/`phasespace` path. After that, the dynamics roadmap is Gounaris-Sakurai for `rho(770)`, Flatte for `f0(980)`, then LASS and K-matrix.

## Reference

J. Back et al., **Laura++: a Dalitz plot fitter**, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.
