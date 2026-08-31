# Square Dalitz integration

`SquareDalitzGrid` provides deterministic integration on the square-Dalitz variables commonly used in three-body amplitude analyses.

For a selected two-body pair `(i, j)`, with invariant mass `m_ij`, the coordinates follow the Laura++ convention

```text
m'     = acos(2 (m_ij - m_min)/(m_max - m_min) - 1) / pi
theta' = theta_ij / pi
```

with

```text
m_min = m_i + m_j
m_max = M - m_k
```

and `k` the bachelor index. Both transformed coordinates lie in `[0, 1]`.

The map is not equal-area in the original Dalitz invariants. `SquareDalitzGrid.sample()` therefore stores the absolute transformation Jacobian in `PhaseSpaceSample.weights`. With the package convention

```text
integral(f) = mean(sample.weights * f)
```

the physical integral is

```text
integral_DP f(s12,s13,s23) ds_a ds_b
=
integral_0^1 integral_0^1 f(s(m',theta')) |J| dm' dtheta'.
```

For the implemented convention,

```text
|J| = 2 pi^2 Delta_m m_ij q p sin(pi m') sin(pi theta')
```

where `q` is the daughter momentum in the `ij` rest frame and `p` is the bachelor momentum in that frame.

## Quadrature

Two deterministic quadratures are available:

```python
quadrature="midpoint"
quadrature="gauss-legendre"
```

The **default** is `midpoint`. It uses a regular, uniform grid of cell centers in both transformed variables,

```text
m'_a     = (a + 1/2) / N
theta'_b = (b + 1/2) / N
```

so the sampling itself is uniform on `[0,1] x [0,1]`; all non-uniformity of the physical measure enters through the Jacobian.

`gauss-legendre` remains available explicitly for convergence studies. It often converges faster for smooth functions, but narrow or highly localized amplitudes still require sufficient resolution. The quadrature choice therefore does not replace a convergence study of the normalization matrix.

The Gauss-Legendre weights are folded into `PhaseSpaceSample.weights` together with the physical Jacobian. They are scaled so that the package-wide convention `mean(weights * f)` remains valid.

## Usage

The standard Square-Dalitz grid is simply

```python
from dalitzplotfitter import SquareDalitzGrid

normalization_sample = SquareDalitzGrid(
    model.channel.parent_mass,
    model.channel.daughter_masses,
    resolution=1000,
    pair=(0, 2),
    quadrature="midpoint",
).sample()

cache = model.prepare_cache(
    data_sample,
    normalization_sample=normalization_sample,
)
```

The same sample may be supplied to `DecayModel.pdf(normalization_sample=...)`.

No amplitude, coefficient, cache, likelihood or minimizer code depends on the coordinate system used to construct the normalization sample. The normalization matrix remains

```text
M_ij = integral F_i^* F_j dPhi
```

and a coherent model is normalized as `c^dagger M c`.

## What must converge in a fit

A comparison of only the total truth normalization

```text
I = c^dagger M c
```

is not sufficient to validate a quadrature. Different errors in matrix elements can cancel for one particular coefficient vector. For fit closure the relevant numerical object is the full complex normalization matrix

```text
M_ij = integral F_i^* F_j dPhi.
```

A robust convergence study should therefore compare all diagonal and interference elements against a denser reference grid. This is especially important for narrow resonances and for interference between structures oriented along different Dalitz axes.

## B+ -> K+ pi+ pi- convention

For particle ordering

```text
(1, 2, 3) = (K+, pi+, pi-)
```

`notebooks/12_cp_coefficients_closure.ipynb` uses

```python
pair=(0, 2)
```

which corresponds to the `(1,3)` pair in one-based notation and therefore transforms

```text
m_13 = m(K+ pi-).
```

The Square Dalitz sample is used for component normalization, the charge integrals `I+` and `I-`, and the joint CP likelihood denominator `I+ + I-`.

## Validation

`tests/test_square_dalitz.py` checks:

- invariant -> square-Dalitz -> invariant round trips;
- the midpoint integral of a constant against the ordinary Dalitz area;
- Gauss-Legendre constant and smooth-moment integrals;
- convergence for a narrow Breit-Wigner-like structure at sufficiently high resolution.

The narrow-structure test intentionally compares a 250x250 Gauss grid against a 500x500 reference at tight relative precision. A coarser 120x120 grid is not expected to reach the same tolerance for a K*(892)-like width. These comparisons protect the transformation, Jacobian and quadrature-weight convention from silent errors.
