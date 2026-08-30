# Resonance dynamics and angular terms

DalitzPlotFitter implements resonance dynamics directly in JAX. Laura++ is one of the principal references used to define and validate the conventions.

## Available lineshapes

The public lineshapes are:

```python
RelativisticBreitWigner()
Pole()
GounarisSakurai()
Flatte(...)
LASS(...)
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

## Simple Pole

`Pole` exposes the simple fixed-width Breit-Wigner form listed as `BW` in Laura++ Appendix A, Eq. (37):

```text
R(m) = 1 / (m - m0 - i Gamma0/2).
```

This is distinct from `RelativisticBreitWigner`: it has no running width and no momentum dependence inside the propagator. The class is named `Pole` in DalitzPlotFitter to make the fixed complex-pole interpretation explicit.

Example:

```python
Resonance(
    "broad_scalar",
    pair=(0, 1),
    coefficient=RealImag(x, y),
    mass=0.478,
    width=0.324,
    spin=0,
    lineshape=Pole(),
)
```

This should not be confused with the dedicated Laura++ `Sigma`/`Kappa` parameterisation of Eqs. (47)-(48), which is a different mass-dependent scalar model and is not implemented yet.

## Gounaris-Sakurai

`GounarisSakurai` follows Laura++ Appendix A, Eqs. (39)-(43), for rho-like vector states:

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

`GounarisSakurai` requires a spin-1 `ResonanceContext`. The pion mass entering the analytic GS function is taken as the arithmetic mean of the two resonance-daughter pion masses.

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

DalitzPlotFitter provides Laura++ Table A.2 presets:

```python
Flatte.f0_980()
Flatte.k0star_1430_neutral()
Flatte.k0star_1430_charged()
Flatte.a0_980_neutral()
Flatte.a0_980_charged()
```

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

For a Flatte component the ordinary `Resonance.width` is not the physical width parameter of the lineshape; the imaginary part is determined by the channel couplings.

## LASS K-pi S-wave

`LASS` follows Laura++ Appendix A, Eqs. (50)-(51):

```text
R(m) = m / [q cot(delta_B) - i q]
     + exp(2 i delta_B)
       [m0 Gamma0 (m0/q0)] /
       [(m0^2-m^2) - i m0 Gamma0 (q/m)(m0/q0)],

cot(delta_B) = 1/(a q) + r q/2.
```

The default effective-range values are the commonly used LASS measurements

```text
a = 2.07 GeV^-1
r = 3.32 GeV^-1.
```

The slowly varying non-resonant part can be cut off with `cutoff`. The resonant term is not removed by that cutoff.

The Laura++ decomposition is available through the `mode` option:

```python
LASS(mode="full")          # Laura++ LASS
LASS(mode="resonant")      # Laura++ LASS_BW
LASS(mode="nonresonant")   # Laura++ LASS_NR
```

or directly with

```python
nonresonant, resonant = LASS().terms(mass, context)
```

Example:

```python
Resonance(
    "Kpi_S",
    pair=(0, 1),
    coefficient=RealImag(x, y),
    mass=1.425,
    width=0.270,
    spin=0,
    lineshape=LASS(scattering_length=2.07, effective_range=3.32, cutoff=1.7),
)
```

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

## Validation notebook

`notebooks/08_lineshape_validation_gs_flatte.ipynb` contains isolated Flatte, Gounaris-Sakurai, Pole and LASS models, deterministic Dalitz-grid densities and 100k-event toy-MC samples for visual validation.

## Reference

J. Back et al., *Laura++: a Dalitz plot fitter*, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854. The Pole/BW, GS, Flatte and LASS equations are taken from Appendix A.
