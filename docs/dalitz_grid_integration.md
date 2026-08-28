# Deterministic Dalitz-grid integration

DalitzPlotFitter provides `DalitzGrid` as a deterministic alternative to Monte Carlo normalization for three-body amplitudes.

The purpose of this mode is to integrate directly in the physical Dalitz variables

```text
s12 = m12^2
s13 = m13^2
```

where the three-body phase-space measure is constant up to an overall channel-dependent factor:

```text
dPhi3 proportional to ds12 ds13.
```

That overall constant cancels in normalized amplitude fits, so the integration can be performed directly as a two-dimensional midpoint quadrature.

## Construction

```python
from dalitzplotfitter import DalitzGrid

norm = DalitzGrid(
    channel.parent_mass,
    channel.daughter_masses,
    resolution=800,
).sample()
```

A resolution of `N` constructs an `N x N` Cartesian grid over the global `(s12, s13)` bounding rectangle. Each point is placed at the centre of its cell. Points whose centres fall outside the exact physical Dalitz boundary are discarded.

The physical limits at fixed `s12` are calculated from

```text
E1* = (s12 + m1^2 - m2^2) / (2 sqrt(s12))
E3* = (M^2 - s12 - m3^2) / (2 sqrt(s12))
q   = sqrt(lambda(s12,m1^2,m2^2)) / (2 sqrt(s12))
p   = sqrt(lambda(M^2,s12,m3^2)) / (2 sqrt(s12))

s13_min = m1^2 + m3^2 + 2(E1* E3* - q p)
s13_max = m1^2 + m3^2 + 2(E1* E3* + q p)
```

and

```text
s23 = M^2 + m1^2 + m2^2 + m3^2 - s12 - s13.
```

## Quadrature weights

All retained cells have the same area

```text
cell_area = Delta_s12 Delta_s13.
```

There is no importance sampling and no event-dependent phase-space weight.

The package integration API uses estimators of the form

```text
mean(weights * f).
```

Therefore a grid sample stores the same constant value for every retained point,

```text
weight = N_valid * cell_area,
```

so that

```text
mean(weight * f)
= cell_area * sum(f),
```

which is exactly the midpoint quadrature.

For one amplitude component,

```text
I_i = integral |F_i|^2 ds12 ds13
```

is approximated by

```text
I_i ~= cell_area * sum_k |F_i(s12_k,s13_k)|^2.
```

For the interference matrix,

```text
M_ij ~= cell_area * sum_k conj(F_i) F_j.
```

The total coherent normalization remains

```text
N = c^dagger M c.
```

## Why use the grid for diagnostics?

A fixed grid removes two numerical ingredients from the likelihood normalization:

```text
random normalization-sample fluctuations
importance-sampling weights
```

For a fixed grid resolution, the normalization is a deterministic function of every floating coefficient and dynamics parameter. This makes the method useful for closure tests and for diagnosing the propagation

```text
Parameter -> amplitude -> normalization -> JAX gradient -> Minuit.
```

The main limitation is discretization. Narrow structures require a sufficiently fine grid. Grid convergence should therefore be checked explicitly by comparing, for example,

```text
300 x 300
500 x 500
800 x 800
```

without changing any other part of the model.

## Current examples

`notebooks/02_fit_dynamic_parameters.ipynb` and `notebooks/03_lineshape_parameter_diagnostics.ipynb` currently use an `800 x 800` deterministic grid for normalization diagnostics.

The pseudo-data candidate pool remains independently generated phase space. Its proposal weights are used only when drawing pseudo-data from the candidate pool. Both the truth intensity used for generation and the fitted likelihood are normalized using the same deterministic Dalitz grid.

The model-owned Monte Carlo normalization remains available and has not been removed or silently replaced as the `DecayModel` default. This keeps the grid comparison isolated from other changes to the fitting architecture.
