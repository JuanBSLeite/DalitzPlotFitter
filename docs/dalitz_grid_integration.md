# Deterministic equal-area Dalitz-grid integration

DalitzPlotFitter provides `DalitzGrid` as a deterministic alternative to Monte Carlo normalization for three-body amplitudes.

The integration is performed directly in

```text
s12 = m12^2
s13 = m13^2
```

where, apart from an overall channel-dependent constant that cancels in normalized amplitude fits,

```text
dPhi3 proportional to ds12 ds13.
```

## Equal-area contour-adapted construction

```python
from dalitzplotfitter import DalitzGrid

norm = DalitzGrid(
    channel.parent_mass,
    channel.daughter_masses,
    resolution=800,
).sample()
```

A resolution of `N` now returns exactly `N**2` physical points. There is no bounding rectangle and no rejection/mask step.

At fixed `s12`, define the physical width

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

We define an auxiliary coordinate

```text
u(s12) = [integral from s12_min to s12 W(s) ds] / A_DP.
```

A regular midpoint grid is built in

```text
u_i = (i + 1/2)/N
v_j = (j + 1/2)/N.
```

The cumulative-area relation is inverted numerically to obtain `s12(u_i)`. The second coordinate is then

```text
s13(u,v) = s13_min(s12) + v * W(s12).
```

Finally,

```text
s23 = M^2 + m1^2 + m2^2 + m3^2 - s12 - s13.
```

All generated grid points are strictly inside the physical Dalitz boundary.

## Constant Jacobian and equal areas

The mapping is constructed so that

```text
ds12/du = A_DP / W(s12)
partial s13/partial v = W(s12).
```

Therefore

```text
|J| = A_DP,
```

and

```text
ds12 ds13 = A_DP du dv.
```

Every cell of the regular `u,v` grid therefore corresponds to the same physical Dalitz area

```text
A_DP / N^2.
```

There is no importance sampling and no event-dependent phase-space weight.

The package cache/integration convention is

```text
mean(weights * f).
```

Therefore every grid point stores the same constant value

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
I_i ~= A_DP * mean(|F_i|^2).
```

For the interference matrix,

```text
M_ij ~= A_DP * mean(conj(F_i) F_j).
```

The coherent normalization remains

```text
N = c^dagger M c.
```

## Numerical inversion of cumulative area

`DalitzGrid` tabulates `W(s12)` on a dense one-dimensional support, integrates it with the trapezoidal rule, and uses linear interpolation to invert cumulative area. The default support is at least 4097 points and grows with the requested two-dimensional resolution.

For dedicated convergence studies this support can be controlled explicitly with

```python
DalitzGrid(..., resolution=800, boundary_resolution=20001)
```

The one-dimensional boundary table is deterministic and is independent of the `N x N` midpoint grid used to evaluate amplitudes.

## Why use this grid for fit diagnostics?

For fixed `N`, normalization is completely deterministic. The method removes

```text
random normalization-sample fluctuations
importance-sampling weights
rejected grid cells
```

from the likelihood normalization.

This is useful for diagnosing the full chain

```text
Parameter -> amplitude -> component normalization
          -> interference matrix -> JAX gradient -> Minuit.
```

The remaining numerical approximation is deterministic quadrature/discretization. Convergence should be checked by comparing, for example,

```text
300 x 300
500 x 500
800 x 800.
```

## Dynamics-aware adaptive refinement

For very narrow structures, a globally regular equal-area grid may spend too few points in the invariant-mass direction near a kinematic boundary. `AdaptiveDalitzGrid` provides an experimental hierarchical refinement scheme for this situation.

```python
from dalitzplotfitter import AdaptiveDalitzGrid

adaptive = AdaptiveDalitzGrid(
    channel.parent_mass,
    channel.daughter_masses,
    base_resolution=48,
    max_depth=5,
    tolerance=0.08,
)

result = adaptive.build((dynamics_probe,))
norm = result.sample
```

The algorithm remains in the same auxiliary `(u,v)` coordinates as `DalitzGrid`, so the physical Jacobian is still the constant `A_DP`. A leaf cell with auxiliary size `du * dv` represents physical area

```text
A_cell = A_DP * du * dv.
```

Because the package estimator is `mean(weights * f)`, a sample with `N_leaf` adaptive cells stores

```text
weight_i = N_leaf * A_cell_i.
```

This gives

```text
mean(weights * f) = sum_i A_cell_i * f_i.
```

Thus unequal adaptive cells integrate with the same `PhaseSpaceSample` convention already used elsewhere in the package.

### Generic refinement criterion

The adaptive grid does not assume a Breit-Wigner or require parameters named `mass` and `width`. Each probe is simply a callable

```python
dynamics_probe(data) -> array
```

on Dalitz points. Complex probes are converted internally to `abs(probe)**2` for the refinement estimator. Each cell is evaluated at its centre and at the four would-be child midpoints. Refinement is triggered by the larger of

```text
local variation across those samples
centre estimate versus four-child midpoint estimate
```

relative to a local scale. Multiple probes may be supplied, and a cell is refined if any probe requires it. This allows the same machinery to be used for narrow resonances, dispersive amplitudes, K-matrix components, splines, tabulated amplitudes, or future user-defined dynamics.

### Discovery-scale limitation

Adaptive refinement cannot discover an arbitrarily narrow feature that falls between all probe points of the initial grid. `base_resolution` therefore remains a physically important control parameter. It defines the scale on which narrow structures are first discovered; recursive refinement controls how accurately they are resolved after discovery.

For this reason, convergence studies should vary both

```text
base_resolution
max_depth / tolerance
```

rather than only the final number of leaf cells.

### Narrow-phi diagnostic

`notebooks/05_adaptive_grid_phi_B2KKK.ipynb` tests the adaptive scheme on

```text
B+ -> K- K+ K+
phi(1020) -> K+ K-
```

where the phi is narrow and close to the `K+K-` threshold. The notebook compares local point density and the convergence of the raw `|F_phi|^2` integral against regular equal-area grids.

## Current examples

`notebooks/02_fit_dynamic_parameters.ipynb` and `notebooks/03_lineshape_parameter_diagnostics.ipynb` use `DalitzGrid`; because the class itself now implements the equal-area contour mapping, those notebooks automatically use exactly `N**2` physical integration points.

`notebooks/04_normalization_grid_diagnostics.ipynb` visualizes the equal-area mapping and quadrature weights. `notebooks/05_adaptive_grid_phi_B2KKK.ipynb` explores the experimental dynamics-aware adaptive grid.

The pseudo-data candidate pool is still independently generated phase space. Its proposal weights are used only when drawing pseudo-data from that pool. Both the truth intensity used for generation and the fitted likelihood are normalized with the same deterministic Dalitz grid.

The model-owned Monte Carlo normalization remains available and has not been replaced as the `DecayModel` default, so the grid studies remain isolated from other changes to the fitting architecture.
