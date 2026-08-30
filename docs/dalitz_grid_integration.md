# Deterministic equal-area Dalitz-grid integration

DalitzPlotFitter supports two normalization strategies for three-body amplitudes:

1. **uniform phase-space Monte Carlo**, through `PhaseSpaceMC` / the model-owned normalization sample;
2. **deterministic equal-area Dalitz grid**, through `DalitzGrid`.

No adaptive quadrature is part of the supported API.

## Monte Carlo normalization

The default model-owned normalization sample is generated with the pure-JAX three-body phase-space generator. The package estimator uses

```text
mean(weights * f)
```

with event-dependent phase-space weights returned by `PhaseSpaceMC`.

Typical use:

```python
model = DecayModel(
    channel,
    components,
    normalization_size=1_000_000,
    normalization_seed=2027,
)

norm = model.normalization_sample
```

For coherent normalization,

```text
M_ij ~= mean(w_PS * conj(F_i) F_j)
N = c^dagger M c.
```

This is the default normalization path of `DecayModel`.

## Equal-area Dalitz-grid normalization

`DalitzGrid` is the deterministic alternative. The integration is performed directly in

```text
s12 = m12^2
s13 = m13^2
```

where, apart from an overall channel-dependent constant that cancels in normalized amplitude fits,

```text
dPhi3 proportional to ds12 ds13.
```

Typical use:

```python
from dalitzplotfitter import DalitzGrid

norm = DalitzGrid(
    channel.parent_mass,
    channel.daughter_masses,
    resolution=800,
).sample()
```

A resolution of `N` returns exactly `N**2` physical points. There is no bounding rectangle and no rejection/mask step.

## Equal-area contour-adapted construction

At fixed `s12`, define

```text
W(s12) = s13_max(s12) - s13_min(s12).
```

The physical limits are

```text
E1* = (s12 + m1^2 - m2^2) / (2 sqrt(s12))
E3* = (M^2 - s12 - m3^2) / (2 sqrt(s12))
q   = sqrt(lambda(s12,m1^2,m2^2)) / (2 sqrt(s12))
p   = sqrt(lambda(M^2,s12,m3^2)) / (2 sqrt(s12))

s13_min = m1^2 + m3^2 + 2(E1* E3* - q p)
s13_max = m1^2 + m3^2 + 2(E1* E3* + q p).
```

The total Dalitz area is

```text
A_DP = integral W(s12) ds12.
```

Define

```text
u(s12) = [integral from s12_min to s12 W(s) ds] / A_DP.
```

A regular midpoint grid is built in

```text
u_i = (i + 1/2)/N
v_j = (j + 1/2)/N.
```

The cumulative-area relation is inverted numerically to obtain `s12(u_i)`, then

```text
s13(u,v) = s13_min(s12) + v * W(s12),
```

and

```text
s23 = M^2 + m1^2 + m2^2 + m3^2 - s12 - s13.
```

All grid points lie inside the physical Dalitz boundary.

## Constant Jacobian and equal areas

The mapping satisfies

```text
ds12/du = A_DP / W(s12)
partial s13/partial v = W(s12),
```

therefore

```text
|J| = A_DP
```

and

```text
ds12 ds13 = A_DP du dv.
```

Every cell represents the same physical area

```text
A_DP / N^2.
```

The package estimator remains

```text
mean(weights * f).
```

Every grid point therefore stores the same constant weight

```text
weight = A_DP,
```

so that

```text
mean(weight * f) = A_DP * mean(f)
                 = (A_DP/N^2) * sum(f).
```

For one amplitude component,

```text
I_i ~= A_DP * mean(|F_i|^2),
```

and for the interference matrix,

```text
M_ij ~= A_DP * mean(conj(F_i) F_j).
```

The coherent normalization is

```text
N = c^dagger M c.
```

## Numerical inversion of cumulative area

`DalitzGrid` tabulates `W(s12)` on a dense one-dimensional support, integrates it with the trapezoidal rule, and uses linear interpolation to invert cumulative area. The default support is at least 4097 points and grows with the requested two-dimensional resolution.

For convergence studies:

```python
DalitzGrid(..., resolution=800, boundary_resolution=20001)
```

The one-dimensional boundary table is deterministic and independent of the `N x N` midpoint grid used to evaluate amplitudes.

## Choosing the normalization method

Use **Monte Carlo normalization** when you want the model default, straightforward scaling to large samples, or a stochastic integration sample generated directly in JAX.

Use **DalitzGrid normalization** when you want a deterministic closure test, direct quadrature-convergence studies, or to remove normalization-sample fluctuations from a fit diagnostic.

For either method, convergence should be checked explicitly by increasing the MC sample size or the grid resolution.

## Current examples

`notebooks/02_fit_dynamic_parameters.ipynb` uses `DalitzGrid` for a deterministic coefficient-closure test.

`notebooks/03_lineshape_parameter_diagnostics.ipynb` and `notebooks/04_normalization_grid_diagnostics.ipynb` provide additional grid and normalization diagnostics.

The model-owned Monte Carlo normalization remains available through `DecayModel.normalization_sample` and is the default normalization path of the high-level model.
