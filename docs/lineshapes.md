# Resonance dynamics and angular terms

DalitzPlotFitter implements resonance dynamics directly in JAX. Laura++ and published LHCb amplitude analyses are principal references used to define and validate the conventions.

## Available dynamics models

```python
RelativisticBreitWigner()
Pole()
GounarisSakurai()
Flatte(...)
LASS(...)
KMatrix(...)
QMI(...)
```

A scalar dynamics plugin is any callable with the interface

```python
lineshape(mass, context)
```

and can be supplied through `Resonance(..., lineshape=...)` without changing the model or cache architecture.

## Relativistic Breit-Wigner

```text
R(m) = 1 / (m0^2 - m^2 - i m0 Gamma(m))
Gamma(m) = Gamma0 (q/q0)^(2L+1) (m0/m) X_L(q r)^2.
```

## Simple Pole

`Pole` exposes the simple fixed-width Breit-Wigner form listed as `BW` in Laura++ Appendix A:

```text
R(m) = 1 / (m - m0 - i Gamma0/2).
```

## Gounaris-Sakurai

`GounarisSakurai` follows Laura++ Appendix A for rho-like vector states,

```text
R(m) = [1 + D Gamma0/m0] /
       [m0^2 - m^2 + f(m) - i m0 Gamma(m)].
```

## Flatte

`Flatte` follows the coupled two-channel Laura++ form

```text
R(m) = 1 / [(m0^2-m^2) - i m0 (Gamma1(m)+Gamma2(m))].
```

with analytic continuation below threshold and optional Adler zero. Presets are provided for `f0(980)`, `K0*(1430)` and `a0(980)` charge states.

## LASS K-pi S-wave

`LASS` implements the coherent effective-range plus `K0*(1430)` S-wave form,

```text
R(m) = m / [q cot(delta_B) - i q]
     + exp(2 i delta_B)
       [m0 Gamma0 (m0/q0)] /
       [(m0^2-m^2) - i m0 Gamma0 (q/m)(m0/q0)],

cot(delta_B) = 1/(a q) + r q/2.
```

## Five-channel K-matrix pi-pi S-wave

`KMatrix` implements the standard five-pole, five-channel Anisovich-Sarantsev model used by Laura++. The channel order is `pi pi`, `K Kbar`, `4 pi`, `eta eta`, `eta eta'`.

```text
K_ij(s) = [ sum_alpha g_i^alpha g_j^alpha/(m_alpha^2-s)
          + f_ij^scatt (1-s0_scatt/s)/(s-s0_scatt) ] f_A0(s)

P_j(s) = sum_alpha beta_alpha g_j^alpha/(m_alpha^2-s)
       + f_1j^prod (1-s0_prod/s)/(s-s0_prod)

F = (I - i K rho)^(-1) P.
```

The scattering constants are fixed by default while the process-dependent `betas` and `f_prod` may be complex fit parameters. `scattering_amplitude()` and `s_matrix()` expose the coupled-channel `T` and `S` matrices for unitarity diagnostics.

## QMI / QMIPWA S-wave

`QMI` implements a quasi-model-independent scalar amplitude specified at fixed two-body mass knots. The interpolation coordinate is

```text
s = m(pi pi)^2.
```

Thus

```text
A_S(s_k) = a_k exp(i delta_k)
A_S(s)   = a(s) exp(i delta(s)).
```

Magnitude and phase are interpolated separately. Two interpolation modes are available:

```python
QMI(..., interpolation="linear")  # default; reproduces the published LHCb convention
QMI(..., interpolation="cubic")   # natural cubic spline in s=m^2
```

The cubic mode is implemented directly in JAX with natural boundary conditions, so the knot magnitudes and phases remain differentiable fit parameters. Both modes pass exactly through all supplied knots; outside the knot range the nearest endpoint value is used.

The public `knots` argument is given as masses in GeV, matching the published tables; internally `QMI` squares the masses and interpolates in `s`. Entries of `magnitudes` and `phases` may be numerical constants or fit `Parameter` objects. Phases are expressed in radians and should be supplied as a continuous/unwrapped sequence; interpolation does not impose a `[-pi, pi)` branch cut.

Example:

```python
qmi = QMI(
    knots=(0.30, 0.50, 0.70, 0.90, 1.10),
    magnitudes=(a0, a1, a2, a3, a4),
    phases=(d0, d1, d2, d3, d4),
    interpolation="cubic",
)

Resonance(
    "pipi_S_qmi",
    pair=(0, 1),
    coefficient=RealImag(1.0, 0.0),
    mass=1.0,
    width=0.0,
    spin=0,
    lineshape=qmi,
)
```

For a pure QMI component the common `Resonance.mass` and `width` fields are placeholders; the shape is entirely defined by the knot values. At least one magnitude/phase convention must be fixed in a fit to remove the overall scale/phase ambiguity of the QMI amplitude. Magnitude parameters should normally be constrained to non-negative values.

`notebooks/10_qmi_validation.ipynb` contains all 50 central values from Table 9 of the published LHCb `D_s+ -> pi- pi+ pi+` analysis. The phases are stored exactly as published in degrees and are unwrapped only before conversion to the continuous interpolation used by `QMI`. The same notebook builds the corresponding 98-parameter fit declaration by fixing one reference magnitude and phase.

## Component composition and normalization

`ResonanceAmplitude` multiplies

```text
lineshape * parent Blatt-Weisskopf * resonance Blatt-Weisskopf * angular factor.
```

Identical final-state particles are symmetrized automatically. All amplitude-component and coherent-PDF normalization uses deterministic `DalitzGrid`; `PhaseSpaceMC` is retained only for toy/proposal generation.

## Validation notebooks

- `notebooks/08_lineshape_validation_gs_flatte.ipynb`: Flatte, Gounaris-Sakurai, Pole and LASS.
- `notebooks/09_kmatrix_validation.ipynb`: K-matrix, coupled-channel unitarity, Dalitz density and toy MC.
- `notebooks/10_qmi_validation.ipynb`: published 50-point `D_s+ -> pi- pi+ pi+` QMI S-wave, interpolation in `s`, Argand trajectory, deterministic Dalitz density, 100k-event toy MC and the full 98-parameter fit declaration.

## References

J. Back et al., *Laura++: a Dalitz plot fitter*, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.

V. V. Anisovich and A. V. Sarantsev, *K-matrix analysis of the (IJ^PC = 00++)-wave in the mass region below 1900 MeV*, Eur. Phys. J. A 16 (2003) 229.

LHCb Collaboration, *Amplitude analysis of the D_s+ -> pi- pi+ pi+ decay*, arXiv:2209.09840.
