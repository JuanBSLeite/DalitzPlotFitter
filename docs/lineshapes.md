# Laura++ resonance dynamics and angular terms

DalitzPlotFitter is moving toward project-owned amplitude dynamics that follow the conventions used by Laura++. The intended physics model no longer relies on AmpForm for the resonance angular term: the resonance lineshape, momentum-dependent width, Blatt-Weisskopf factors and angular factor are all defined explicitly by DalitzPlotFitter according to Laura++ conventions.

## Architecture

The intended model flow is

```text
particle masses / decay definition
  -> weighted phase-space Monte Carlo
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
  -> cached likelihood and normalization
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

DalitzPlotFitter provides

```python
from dalitzplotfitter.dynamics import covariant_angular_factor

T = covariant_angular_factor(
    p_star=p_star,
    p=p,
    q=q,
    cos_theta=cos_theta,
    parent_mass=m_parent,
    angular_momentum=L,
)
```

The implemented expressions are Eqs. (91)-(95) of J. Back et al., *Laura++: a Dalitz plot fitter*, CPC 231 (2018) 198-242, for `L=0..4`.

They are

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

The helper functions

```python
bachelor_momentum_parent_frame(...)
bachelor_momentum_resonance_frame(...)
breakup_momentum(...)
```

provide the corresponding `p*`, `p` and `q` magnitudes from masses.

The helicity-angle sign depends on which resonance daughter is chosen. The final three-body component implementation must use the same daughter-ordering convention as Laura++, and identical-particle symmetrisation must apply the corresponding angular term separately to each pairing.

## Laura++ relativistic Breit-Wigner

The first native resonance lineshape implementation is the Laura++-convention relativistic Breit-Wigner

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

The implementation currently supports orbital angular momenta `L=0..5` for the Laura++ Blatt-Weisskopf polynomials.

## Complete resonance component

For an isobar resonance component the target convention is

```text
F_i(x) = R_i(m) X_parent X_resonance T_L(x),
```

with every factor following Laura++ conventions. For identical particles the complete amplitude is the coherent sum over the required pairings before multiplication by the external complex coefficient.

This complete component, rather than only `R(m)`, is what enters the fit and the Monte Carlo normalization.

## Monte Carlo normalization

The planned phase-space backend is the `phasespace` package, which supplies four-momenta and phase-space event weights. The normalization of a coefficient-only amplitude is then estimated using a fixed weighted sample:

```text
M_ij proportional to sum_k w_PS,k F_i*(x_k) F_j(x_k),
N(c) = c^dagger M c.
```

The normalization sample is generated once and remains fixed during minimization.

For the current high-statistics closure benchmark the intended sample sizes are

```text
fit sample:           100,000 events
normalization sample: 1,000,000 events
```

## Individual component normalization

If individual component normalization is enabled, it must apply to the **complete Laura++ component** including

- resonance lineshape;
- Covariant angular term;
- Blatt-Weisskopf factors;
- identical-particle symmetrisation;
- any other dynamics included in that component.

The precise convention concerning efficiency in this basis normalization will be validated separately before being treated as production-ready.

## Planned native dynamics

The implementation sequence is:

1. Laura++ relativistic Breit-Wigner — implemented;
2. Laura++ Covariant angular formalism — implemented symbolically for `L=0..4`;
3. weighted phase-space MC backend using `phasespace` — planned;
4. combine RBW, Blatt-Weisskopf and Covariant terms into the full resonance component;
5. Gounaris-Sakurai — planned, especially for `rho(770)`;
6. Flatte — planned, especially for `f0(980)` near the `K Kbar` threshold;
7. LASS — planned for `K pi` S-wave models;
8. K-matrix after the simpler models have closure and reference-validation tests.

## Validation requirements

Every physics term should have unit tests for its analytic limits and numerical comparison tests against the Laura++ convention. Existing checks include

```text
q(m0) = q0
Gamma(m0) = Gamma0
X(q0 r) = 1
T_0 = 1
```

and direct numerical tests of the published Covariant expressions for `L=1..4`.

The next important validation is to construct the full `rho(770)` component with the Covariant angular term and compare it point-by-point against Laura++ using identical Dalitz kinematics and daughter ordering.

## Reference

J. Back et al., **Laura++: a Dalitz plot fitter**, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.
