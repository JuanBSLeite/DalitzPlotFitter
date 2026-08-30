# Resonance dynamics and angular terms

DalitzPlotFitter implements resonance dynamics directly in JAX. Laura++ is one of the principal references used to define and validate the conventions.

## Available lineshapes

The public lineshapes are:

```python
RelativisticBreitWigner()
GounarisSakurai()
Flatte(...)
```

A lineshape is any callable with the interface

```python
lineshape(mass, context)
```

and can be supplied through `Resonance(..., lineshape=...)` without changing the model or cache architecture.

## Relativistic Breit-Wigner

The default is

```text
R(m) = 1 / (m0^2 - m^2 - i m0 Gamma(m))
```

with

```text
Gamma(m) = Gamma0 (q/q0)^(2L+1) (m0/m) X_L(q r)^2.
```

The Blatt-Weisskopf factors support `L=0..4`.

## Gounaris-Sakurai

`GounarisSakurai` follows Laura++ Appendix A, Eqs. (38)-(43), for rho-like vector states:

```text
R(m) = [1 + D Gamma0/m0] /
       [m0^2 - m^2 + f(m) - i m0 Gamma(m)].
```

The dispersive correction is

```text
f(m) = Gamma0 m0^2/q0^3 *
       {q^2 [h(m)-h(m0)] + (m0^2-m^2) q0^2 dh/dm^2|m0},

h(m) = (2/pi) (q/m) log[(m+2q)/(2 m_pi)].
```

`GounarisSakurai` requires a spin-1 `ResonanceContext`. The pion mass entering the analytic GS function is taken as the arithmetic mean of the two resonance-daughter pion masses, which also permits charged rho decays with the small charged/neutral pion mass difference.

Example:

```python
Resonance(
    "rho(770)0",
    pair=(0, 1),
    coefficient=RealImag(1.0, 0.0),
    lineshape=GounarisSakurai(),
)
```

## Flatte

`Flatte` follows the coupled two-channel form in Laura++ Appendix A, Eqs. (44)-(46):

```text
R(m) = 1 / [(m0^2-m^2) - i m0 (Gamma1(m)+Gamma2(m))].
```

For each channel the width is a coupling times the Laura++ isospin-weighted phase-space factors. Below a specific two-body threshold the square root is analytically continued,

```text
sqrt(1 - m_threshold^2/m^2) -> i sqrt(m_threshold^2/m^2 - 1),
```

so the closed channel contributes to the real part of the denominator and produces the expected threshold cusp.

The optional Adler-zero factor is

```text
f_A = (m^2-s_A)/(m0^2-s_A).
```

Laura++ restricts this model to the systems tabulated in its Table A.2. DalitzPlotFitter provides matching presets:

```python
Flatte.f0_980()
Flatte.k0star_1430_neutral()
Flatte.k0star_1430_charged()
Flatte.a0_980_neutral()
Flatte.a0_980_charged()
```

The preset channel masses, coupling ratios and Adler-zero constants follow that table. The resulting dataclass remains editable, so analyses can construct `Flatte(...)` directly when they need alternative measured couplings.

Example:

```python
Resonance(
    "f0_980",
    pair=(0, 1),
    coefficient=RealImag(x, y),
    mass=0.965,
    width=0.0,
    spin=0,
    lineshape=Flatte.f0_980(),
)
```

For a Flatte component the ordinary `Resonance.width` is not the physical width parameter of the lineshape; the imaginary part is determined by the channel couplings. A numerical non-negative width is still supplied because `ResonanceContext` has a common interface for all lineshapes.

## Component composition

For

```text
P -> R b
R -> d1 d2
```

`ResonanceAmplitude` multiplies

```text
lineshape * parent Blatt-Weisskopf * resonance Blatt-Weisskopf * angular factor.
```

The default angular model is `CovariantAngular()`, and identical final-state particles are symmetrized automatically.

## Deterministic normalization

All amplitude-component and coherent-PDF normalization uses the deterministic equal-area `DalitzGrid`. `PhaseSpaceMC` is retained only for toy/proposal generation.

## Reference

J. Back et al., *Laura++: a Dalitz plot fitter*, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854. The GS and Flatte equations and the Flatte preset systems are taken from Appendix A. citeturn338539search0turn658613search26
