# Resonance dynamics and angular terms

DalitzPlotFitter implements resonance dynamics directly in JAX. Laura++ is one of the principal references used to define and validate the current conventions, but the public API uses neutral physics names rather than reference-specific class names.

## Model flow

```text
DecayChannel + component declarations
  -> particle masses / widths / spins
  -> ResonanceContext
  -> interchangeable lineshape R(m)
  -> Blatt-Weisskopf factors
  -> interchangeable angular model T_L
  -> automatic identical-particle symmetrization
  -> complete component F_i(x)
  -> RealImag coefficient
  -> coherent amplitude
  -> normalized SignalPDF
```

## Particle properties

The high-level `DecayChannel` and `Resonance` API uses the Scikit-HEP `particle` package for standard masses, widths and spins. Values are converted from the database units to GeV before entering the numerical model.

Analysis-specific values are explicit overrides. This is important for historical amplitude models in which the fitted pole parameters do not coincide with current reference values.

## ResonanceContext

`DecayModel` converts the decay channel and resonance declaration into a `ResonanceContext` containing the physical quantities needed by dynamics plugins:

```text
parent mass
resonance-daughter masses
bachelor mass
spin
pole mass
pole width
parent radius
resonance radius
```

Users normally do not construct this object manually.

## Interchangeable lineshapes

A lineshape is any callable with the interface

```python
lineshape(mass, context)
```

The default is

```python
RelativisticBreitWigner()
```

so these are equivalent:

```python
Resonance(
    "rho(770)0",
    pair=(0, 1),
    coefficient=RealImag(1.0, 0.0),
)
```

and

```python
Resonance(
    "rho(770)0",
    pair=(0, 1),
    coefficient=RealImag(1.0, 0.0),
    lineshape=RelativisticBreitWigner(),
)
```

Future dynamics such as Gounaris-Sakurai, Flatte, LASS or K-matrix plug into the same field without changing `DecayModel`.

## Covariant angular formalism

The default angular model is `CovariantAngular()`. For

```text
P -> R b
R -> d1 d2
```

define `p*` as the bachelor momentum in the parent rest frame, `p` as the bachelor momentum in the resonance rest frame, `q` as the selected resonance-daughter momentum in the resonance rest frame, and `theta` as the angle between that daughter and the bachelor in the resonance rest frame.

`covariant_spin_factor()` implements the current convention for `L=0..4`:

```text
T0 = 1
T1 = -2 (p* q) sqrt(1 + p^2/mP^2) cos(theta)
T2 = (4/3) (p* q)^2 (3/2 + p^2/mP^2) [3 cos^2(theta) - 1]
T3 = -(24/15) (p* q)^3 sqrt(1 + p^2/mP^2)
     (5/2 + p^2/mP^2) [5 cos^3(theta) - 3 cos(theta)]
T4 = (16/35) (p* q)^4 [8 p^4/mP^4 + 40 p^2/mP^2 + 35]
     [35 cos^4(theta) - 30 cos^2(theta) + 3]
```

The angular model is also interchangeable. `Resonance(..., angular=...)` can replace the default without coupling the choice to the lineshape.

## Relativistic Breit-Wigner

The default Breit-Wigner lineshape is

```text
R(m) = 1 / (m0^2 - m^2 - i m0 Gamma(m))
```

with

```text
Gamma(m) = Gamma0 (q/q0)^(2L+1) (m0/m) X_L(q r)^2.
```

`RelativisticBreitWigner` uses only `mass` and the fields it needs from `ResonanceContext`.

## High-level resonance declaration

Users should normally construct models through `DecayModel`:

```python
channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
component = Resonance(
    "rho(770)0",
    pair=(0, 1),
    coefficient=RealImag(1.0, 0.0),
)
model = DecayModel(channel, [component])
```

The model builder determines parent mass, resonance-daughter masses, bachelor mass, momentum keys and identical-particle permutations automatically.

## Identical particles

For

```text
D+ -> pi- pi+ pi+
```

a resonance declared with nominal pair `(0,1)` automatically evaluates

```text
F = F[(12)3] + F[(13)2].
```

Only exchanges of identical final-state labels are added. A constant non-resonant term is not duplicated.

## Monte Carlo normalization

`phasespace` supplies raw phase-space weights. For a coefficient-only model,

```text
M_ij = (1/N_MC) sum_k w_PS,k F_i*(x_k) F_j(x_k)
N(c) = c^dagger M c.
```

The reference normalization sample is 1,000,000 weighted events. Pseudo-data are generated from a larger weighted pool with

```text
w_target,k = w_PS,k |A(x_k)|^2
```

and converted to ordinary unweighted events with `weighted_resample()`.

## Validation references

The implementation is checked against analytic limits and established amplitude-analysis conventions. Laura++ is currently a principal reference for Breit-Wigner, Blatt-Weisskopf and covariant angular definitions:

J. Back et al., *Laura++: a Dalitz plot fitter*, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.
