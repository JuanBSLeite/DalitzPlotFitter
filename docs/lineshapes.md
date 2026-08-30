# Resonance dynamics and angular terms

DalitzPlotFitter implements resonance dynamics directly in JAX. Laura++ is one of the principal references used to define and validate the conventions.

## Available lineshapes

The public dynamics models are:

```python
RelativisticBreitWigner()
Pole()
GounarisSakurai()
Flatte(...)
LASS(...)
KMatrix(...)
```

A scalar dynamics plugin is any callable with the interface

```python
lineshape(mass, context)
```

and can be supplied through `Resonance(..., lineshape=...)` without changing the model or cache architecture. `KMatrix` uses the same interface externally, but internally evaluates a coupled five-channel matrix amplitude.

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

`GounarisSakurai` requires a spin-1 `ResonanceContext`.

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

## LASS K-pi S-wave

`LASS` follows Laura++ Appendix A, Eqs. (50)-(51):

```text
R(m) = m / [q cot(delta_B) - i q]
     + exp(2 i delta_B)
       [m0 Gamma0 (m0/q0)] /
       [(m0^2-m^2) - i m0 Gamma0 (q/m)(m0/q0)],

cot(delta_B) = 1/(a q) + r q/2.
```

The default effective-range values are

```text
a = 2.07 GeV^-1
r = 3.32 GeV^-1.
```

The Laura++ decomposition is available through

```python
LASS(mode="full")
LASS(mode="resonant")
LASS(mode="nonresonant")
```

## Five-channel K-matrix pi-pi S-wave

`KMatrix` implements the standard five-pole, five-channel Anisovich-Sarantsev model used by Laura++ for the scalar pi-pi system. The channel order is

```text
1: pi pi
2: K Kbar
3: 4 pi
4: eta eta
5: eta eta'
```

The scattering matrix is

```text
K_ij(s) = [ sum_alpha g_i^alpha g_j^alpha/(m_alpha^2-s)
          + f_ij^scatt (1-s0_scatt/s)/(s-s0_scatt) ] f_A0(s)
```

with Adler factor

```text
f_A0(s) = (1-s_A0/s) (s-s_A m_pi^2/2).
```

The production vector has the same pole structure,

```text
P_j(s) = sum_alpha beta_alpha g_j^alpha/(m_alpha^2-s)
       + f_1j^prod (1-s0_prod/s)/(s-s0_prod),
```

and the physical coupled-channel amplitude is

```text
F = (I - i K rho)^(-1) P.
```

`KMatrix(...)` returns the pi-pi component `F_1` through the ordinary lineshape interface. The five bare masses, the 25 pole couplings, `f_scatt`, `s0_scatt`, `s_A0`, and `s_A` are fixed to the standard scattering solution. Process-dependent production parameters are supplied through the five complex `betas` and five complex `f_prod` values.

For a fitter model, these production quantities can be `RealImag` objects whose real and imaginary parts are `Parameter` instances. They are then collected by `DecayModel.parameters` and can be minimized like other fit parameters, while the scattering constants remain fixed by default.

Example:

```python
kmatrix = KMatrix(
    betas=(
        RealImag(beta1_re, beta1_im),
        RealImag(beta2_re, beta2_im),
        0j,
        0j,
        0j,
    ),
    f_prod=(
        RealImag(f11_re, f11_im),
        0j,
        0j,
        0j,
        0j,
    ),
)

Resonance(
    "pipi_S_kmatrix",
    pair=(0, 1),
    coefficient=RealImag(1.0, 0.0),
    mass=1.0,
    width=0.0,
    spin=0,
    lineshape=kmatrix,
)
```

The `mass` and `width` fields in that declaration are placeholders required by the common `ResonanceContext`; the K-matrix uses its own fixed pole table. Because the component is scalar, the external Blatt-Weisskopf and angular factors are unity.

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

## Validation notebooks

`notebooks/08_lineshape_validation_gs_flatte.ipynb` contains isolated Flatte, Gounaris-Sakurai, Pole and LASS models.

`notebooks/09_kmatrix_validation.ipynb` shows the five K-matrix phase-space channels, the first row of `K(s)`, the P-vector, the rescattered pi-pi amplitude, a deterministic Dalitz-grid model and a 100k-event toy MC.

## References

J. Back et al., *Laura++: a Dalitz plot fitter*, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.

V. V. Anisovich and A. V. Sarantsev, *K-matrix analysis of the (IJ^PC = 00++)-wave in the mass region below 1900 MeV*, Eur. Phys. J. A 16 (2003) 229.
