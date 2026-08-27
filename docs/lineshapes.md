# Laura++ resonance dynamics and angular terms

DalitzPlotFitter is moving toward project-owned amplitude dynamics that follow the conventions used by Laura++. The intended physics model no longer relies on AmpForm for the resonance angular term: the resonance lineshape, momentum-dependent width, Blatt-Weisskopf factors and angular factor are all defined explicitly by DalitzPlotFitter according to Laura++ conventions.

## Architecture

The intended model flow is

```text
particle masses / decay definition
  -> weighted phase-space Monte Carlo from phasespace
  -> Dalitz invariants and four-momenta
  -> DalitzPlotFitter Laura++ dynamics
       - resonance lineshape R(m)
       - running width Gamma(m)
       - Blatt-Weisskopf factors
       - Covariant angular factor T_L
       - identical-particle symmetrisation
  -> complete component F_i(x)
  -> RealImag coefficient c_i = x_i + i y_i
  -> coherent amplitude A = sum_i c_i F_i
  -> cached weighted-MC normalization
  -> optional weighted resampling to unweighted pseudo-data
  -> JAX likelihood + iminuit
```

The physics convention should be explicit and testable term by term against Laura++.

## Covariant angular formalism

The default angular convention for DalitzPlotFitter is the **Laura++ Covariant formalism**, not Zemach or Legendre.

For a decay

```text
P -> R b,
R -> d1 d2,
```

Laura++ defines

```text
p* = bachelor momentum in the P rest frame,
p  = bachelor momentum in the R rest frame,
q  = daughter momentum in the R rest frame,
theta = helicity angle between the selected R daughter and the bachelor in the R rest frame,
mP = parent mass.
```

DalitzPlotFitter provides both symbolic and numerical implementations:

```python
from dalitzplotfitter.dynamics import (
    covariant_angular_factor,
    covariant_spin_factor,
)
```

The expressions are Eqs. (91)-(95) of J. Back et al., *Laura++: a Dalitz plot fitter*, CPC 231 (2018) 198-242, for `L=0..4`.

```text
L = 0:
T_0 = 1

L = 1:
T_1 = -2 (p* q) sqrt(1 + p^2/mP^2) cos(theta)

L = 2:
T_2 = (4/3) (p* q)^2 (3/2 + p^2/mP^2)
      [3 cos^2(theta) - 1]

L = 3:
T_3 = -(24/15) (p* q)^3 sqrt(1 + p^2/mP^2)
      (5/2 + p^2/mP^2)
      [5 cos^3(theta) - 3 cos(theta)]

L = 4:
T_4 = (16/35) (p* q)^4
      [8 p^4/mP^4 + 40 p^2/mP^2 + 35]
      [35 cos^4(theta) - 30 cos^2(theta) + 3]
```

The numerical prefactors and signs are part of the Laura++ amplitude convention and are not absorbed into the fitted coefficient.

`covariant_kinematics(daughter, partner, bachelor)` computes `m`, `p*`, `p`, `q` and `cos(theta)` directly from four-vectors in `(E, px, py, pz)` order. The chosen daughter fixes the odd-spin sign convention. Tests verify that exchanging equal-mass resonance daughters flips `cos(theta)` and the complete `L=1` amplitude sign.

## Laura++ relativistic Breit-Wigner

The first native resonance lineshape implementation is

```text
R(m) = 1 / (m0^2 - m^2 - i m0 Gamma(m))
```

with

```text
Gamma(m) = Gamma0 (q/q0)^(2L+1) (m0/m) X(q r)^2.
```

The two-body breakup momentum is

```text
q(m) = sqrt(lambda(m^2, m1^2, m2^2)) / (2 m),
```

where

```text
lambda(x,y,z) = x^2 + y^2 + z^2 - 2xy - 2xz - 2yz.
```

The Blatt-Weisskopf factor is normalized at the resonance pole so that

```text
X(q0 r) = 1.
```

## Complete resonance component

`LauraCovariantRBW` is the first complete numerical component and evaluates

```text
F_i(x) = R_i(m) X_parent X_resonance T_L(x).
```

It receives raw four-vectors and performs its own Laura++ covariant kinematics. It therefore does not depend on an AmpForm angular factor. The first validation tests cover finite physical values and the daughter-exchange parity of `L=0` and `L=1` components.

For identical particles the production amplitude must coherently sum the corresponding pairings before multiplication by the external complex coefficient.

## Weighted phase-space Monte Carlo

`PhasespaceMC` wraps the external `phasespace` package. The package generates in `(px, py, pz, E)` order; DalitzPlotFitter converts immediately to `(E, px, py, pz)` and then stays in JAX.

For MC integration the wrapper explicitly requests

```text
normalize_weights=False
```

so independent batches retain mutually compatible raw phase-space weights. The coefficient-only normalization matrix is estimated as

```text
M_ij proportional to sum_k w_PS,k F_i*(x_k) F_j(x_k),
N(c) = c^dagger M c.
```

The reference normalization target is **1,000,000 weighted events**.

## Weighted pseudo-data resampling

For an ordinary unbinned closure fit, weighted MC candidates are not inserted into the likelihood as if they were independent unweighted observations. Instead the target candidate weight is

```text
w_target,k = w_PS,k |A(x_k)|^2.
```

`weighted_resample()` draws unweighted pseudo-data from those probabilities. The returned events carry unit weights. The reference pseudo-data target remains **100,000 events**; the candidate pool should be substantially larger than the requested pseudo-data sample.

## Individual component normalization

If individual component normalization is enabled, it must apply to the **complete Laura++ component** including

- resonance lineshape;
- Covariant angular term;
- Blatt-Weisskopf factors;
- identical-particle symmetrisation;
- any other dynamics included in that component.

The precise convention concerning efficiency in this basis normalization will be validated separately before being treated as production-ready.

## Implementation sequence

1. Laura++ relativistic Breit-Wigner — implemented;
2. Laura++ Covariant angular formalism — implemented for `L=0..4`;
3. four-vector covariant kinematics — implemented;
4. weighted `phasespace` MC wrapper — implemented;
5. complete `LauraCovariantRBW` component — implemented, validation in progress;
6. weighted pseudo-data resampling — implemented;
7. migrate the `D+ -> pi- pi+ pi+` closure to the new chain;
8. Gounaris-Sakurai for `rho(770)`;
9. Flatte for `f0(980)`;
10. LASS and K-matrix after the simpler models pass closure/reference tests.

## Validation requirements

Every physics term should have unit tests for its analytic limits and numerical comparison tests against Laura++. Existing checks include

```text
q(m0) = q0
Gamma(m0) = Gamma0
X(q0 r) = 1
T_0 = 1
```

plus direct numerical tests of the Covariant expressions, Lorentz-boost/rest-frame consistency, equal-mass daughter exchange, weighted phase-space conservation and weighted-resampling behavior.

The next reference-level validation is to compare the full `rho(770)` component point-by-point against Laura++ using identical Dalitz kinematics and daughter ordering.

## Reference

J. Back et al., **Laura++: a Dalitz plot fitter**, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.
