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

`QMI` implements a quasi-model-independent scalar amplitude specified directly at invariant-mass knots. The default convention follows the LHCb `D+ -> pi- pi+ pi+` QMIPWA: magnitude and phase are free at each knot and are linearly interpolated separately,

```text
A_S(m_k) = a_k exp(i delta_k)
A_S(m)   = a(m) exp(i delta(m)).
```

`knots` are fixed mass positions in GeV. Entries of `magnitudes` and `phases` may be numerical constants or fit `Parameter` objects. Phases are expressed in radians and should be supplied as a continuous/unwrapped sequence; the interpolation does not impose a `[-pi, pi)` branch cut.

Example:

```python
qmi = QMI(
    knots=(0.30, 0.50, 0.70, 0.90, 1.10),
    magnitudes=(a0, a1, a2, a3, a4),
    phases=(d0, d1, d2, d3, d4),
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

The published LHCb `D+ -> pi- pi+ pi+` QMIPWA uses 50 non-uniform mass knots; `QMI` itself is generic and supports any strictly increasing knot sequence with at least two points.

## Component composition and normalization

`ResonanceAmplitude` multiplies

```text
lineshape * parent Blatt-Weisskopf * resonance Blatt-Weisskopf * angular factor.
```

Identical final-state particles are symmetrized automatically. All amplitude-component and coherent-PDF normalization uses deterministic `DalitzGrid`; `PhaseSpaceMC` is retained only for toy/proposal generation.

## Validation notebooks

- `notebooks/08_lineshape_validation_gs_flatte.ipynb`: Flatte, Gounaris-Sakurai, Pole and LASS.
- `notebooks/09_kmatrix_validation.ipynb`: K-matrix, coupled-channel unitarity, Dalitz density and toy MC.

## References

J. Back et al., *Laura++: a Dalitz plot fitter*, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.

V. V. Anisovich and A. V. Sarantsev, *K-matrix analysis of the (IJ^PC = 00++)-wave in the mass region below 1900 MeV*, Eur. Phys. J. A 16 (2003) 229.

LHCb Collaboration, *Amplitude analysis of the D+ -> pi- pi+ pi+ decay and measurement of the pi- pi+ S-wave amplitude*, JHEP 06 (2023) 044.
