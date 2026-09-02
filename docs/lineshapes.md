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
QMI2D(...)
```

One-dimensional isobar dynamics use the ordinary `lineshape(mass, context)` interface through `Resonance`. A genuinely two-dimensional Dalitz amplitude such as `QMI2D` is attached through `DalitzAmplitude` because it depends simultaneously on two invariant-mass-squared coordinates.

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
```

Published QMI values should be validated in analysis-specific studies before
being used in a production model.

## QMI2D Dalitz amplitude

`QMI2D` is the direct two-dimensional extension of the QMI idea. Every Dalitz cell carries one complex amplitude

```text
A_ij = a_ij exp(i phi_ij).
```

The axes are given directly in Dalitz invariants (`s12` and `s13`) through bin edges. Magnitudes and phases may contain ordinary numbers or dynamical `Parameter` objects, so every active cell can be floated in a fit.

Three evaluation modes are available:

```python
QMI2D(..., interpolation="none")
QMI2D(..., interpolation="linear")
QMI2D(..., interpolation="cubic")
```

- `none` is piecewise constant: every event receives exactly the complex number assigned to its bin.
- `linear` treats the cell values as located at bin centers and interpolates magnitude and phase bilinearly.
- `cubic` performs a local tensor-product bicubic Catmull-Rom interpolation, again separately for magnitude and phase.

All three implementations are JAX-native and therefore compatible with automatic differentiation and minimization.

### Physical Dalitz-bin mask

For a real three-body decay, the rectangular `s12 x s13` grid contains cells that never intersect the physical Dalitz region. `physical_bin_mask(...)` marks only cells with physical support using the exact analytic Dalitz boundary. The bin edges themselves should be built from the exact kinematic endpoints,

```text
s12_min = (m1 + m2)^2
s12_max = (M - m3)^2
```

rather than from the minimum/maximum of numerical integration samples. This is
important because quadrature nodes do not lie exactly on the kinematic
endpoints, while a physical boundary bin must extend all the way to `s12_max`.

Example:

```python
smin = (m1 + m2)**2
smax = (M - m3)**2
edges = np.linspace(smin, smax, 9)

mask = physical_bin_mask(
    tuple(edges), tuple(edges),
    mother_mass=M,
    masses=(m1, m2, m3),
    folded=True,
)

field = QMI2D(
    s12_edges=tuple(edges),
    s13_edges=tuple(edges),
    magnitudes=magnitudes,
    phases=phases,
    active_mask=mask,
    interpolation="cubic",
    folded=True,
)
```

For `interpolation="none"`, inactive cells evaluate to zero. For linear/cubic interpolation, inactive rectangular cells act only as ghost support filled from the nearest active cell; they are not intended to carry independent physics parameters.

For channels with two identical particles, `folded=True` evaluates the field at

```text
s_low  = min(s12, s13)
s_high = max(s12, s13)
```

which imposes the exchange symmetry directly on the two-dimensional field. This is the natural default for studies of `D_s+ -> pi- pi+ pi+` when `s12` and `s13` correspond to the two `pi+ pi-` combinations.

A QMI2D component is attached directly to the coherent amplitude model:

```python
model = DecayModel(
    channel,
    [DalitzAmplitude("qmi2d", field, RealImag(1.0, 0.0))],
)
```

The global complex normalization/phase ambiguity remains present, just as for a 1D QMI, and a fit must fix an appropriate reference convention. A completely free two-dimensional field can also develop poorly constrained or null directions; closure tests and Hessian/correlation diagnostics are therefore essential before using it on data.

## Component composition and normalization

`ResonanceAmplitude` multiplies one-dimensional isobar lineshapes by their barrier and angular terms. `DalitzAmplitude` bypasses the isobar construction and evaluates a full Dalitz-dependent complex function directly.

All amplitude-component and coherent-PDF normalization uses deterministic
mass-plane Gauss--Legendre or Square-Dalitz quadrature; `PhaseSpaceMC` is
retained only for toy/proposal generation.

## References

J. Back et al., *Laura++: a Dalitz plot fitter*, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.

V. V. Anisovich and A. V. Sarantsev, *K-matrix analysis of the (IJ^PC = 00++)-wave in the mass region below 1900 MeV*, Eur. Phys. J. A 16 (2003) 229.

LHCb Collaboration, *Amplitude analysis of the D_s+ -> pi- pi+ pi+ decay*, arXiv:2209.09840.
