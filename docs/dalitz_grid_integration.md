# Deterministic Dalitz-grid integration

DalitzPlotFitter provides deterministic normalization on the ordinary Dalitz plot through both a fixed equal-area `DalitzGrid` and an amplitude-aware `AdaptiveDalitzGrid`. Monte Carlo remains available for event/toy generation through `PhaseSpaceMC`, but it is not the normalization method.

## Uniform equal-area grid

The integration is performed directly in

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
    resolution=1000,
).sample()
```

A resolution `N` returns exactly `N**2` physical points. There is no bounding rectangle and no rejection/mask step.

## Equal-area contour-adapted construction

At fixed `s12`, define

```text
W(s12) = s13_max(s12) - s13_min(s12).
```

The total Dalitz area is

```text
A_DP = integral W(s12) ds12.
```

Define the cumulative-area coordinate

```text
u(s12) = [integral from s12_min to s12 W(s) ds] / A_DP
```

and a second coordinate

```text
v = [s13 - s13_min(s12)] / W(s12).
```

Both lie in `[0,1]`. The inverse map is

```text
s12 = s12(u)
s13 = s13_min(s12) + v W(s12).
```

The Jacobian is constant:

```text
|d(s12,s13)/d(u,v)| = A_DP.
```

Therefore

```text
ds12 ds13 = A_DP du dv.
```

The ordinary `DalitzGrid` uses a uniform midpoint grid in `(u,v)`. Every point is physical and every cell has identical physical area.

## Adaptive ordinary Dalitz grid

`AdaptiveDalitzGrid` uses the same equal-area `(u,v)` map, but recursively subdivides only cells whose amplitude bilinears have not converged.

```python
from dalitzplotfitter import AdaptiveDalitzGrid

adaptive = AdaptiveDalitzGrid(
    channel.parent_mass,
    channel.daughter_masses,
    base_resolution=18,
    min_depth=1,
    max_depth=6,
    tolerance=0.02,
).build(model)

normalization_sample = adaptive.sample
```

For each cell, the algorithm compares a midpoint estimate with the average of four quarter-cell midpoint estimates for the complete raw normalization-matrix integrand

```text
F_i^* F_j.
```

A cell is refined when any numerically relevant matrix element exceeds the requested local tolerance. Because the decision uses the amplitudes themselves rather than resonance metadata, the same mechanism can refine Breit-Wigner, Flatte, LASS, K-matrix, QMI, spline or arbitrary direct Dalitz amplitudes.

`min_depth` forces a minimum number of global subdivisions before the error criterion may stop refinement. This is useful for protecting against ultra-narrow structures that could otherwise fall between the first coarse sample points.

The result stores diagnostics:

```text
adaptive.leaf_bounds
adaptive.leaf_depths
adaptive.leaf_errors
adaptive.u
adaptive.v
```

and remains directly compatible with the existing cache API:

```python
cache = model.prepare_cache(
    data_sample,
    normalization_sample=adaptive.sample,
)
```

## Normalization convention

The package estimator is

```text
mean(weights * f).
```

For the uniform grid every point stores `A_DP`. For the adaptive grid each accepted subcell contributes its own physical quadrature area, rescaled by the total number of returned points so that the same `mean(weights * f)` convention remains valid.

The normalization matrix remains

```text
M_ij = integral F_i^* F_j dPhi
```

and the coherent normalization is

```text
N = c^dagger M c.
```

## Convergence

For production fits, compare the full complex matrix against a denser reference rather than checking only one total normalization. Narrow resonances and rapidly varying interference terms are especially important.

`notebooks/14_adaptive_sqdp_phi_kkk.ipynb` compares, for a narrow `phi(1020)` in `B+ -> K- K+ K+`:

- uniform ordinary Dalitz;
- adaptive ordinary Dalitz;
- uniform Square Dalitz;
- adaptive Square Dalitz.

The preferred method should be chosen from matrix-element accuracy versus number of normalization points, not from the coordinate system alone.

## Current examples

- `notebooks/02_fit_dynamic_parameters.ipynb`: coefficient closure with deterministic grid normalization;
- `notebooks/04_normalization_grid_diagnostics.ipynb`: uniform-grid diagnostics;
- `notebooks/07_e791_rho1450_mass_width_closure.ipynb`: coefficient plus mass/width closure;
- `notebooks/14_adaptive_sqdp_phi_kkk.ipynb`: ordinary/Square, uniform/adaptive comparison for a narrow phi(1020).

`PhaseSpaceMC` is retained for generating event pools and pseudo-data.
