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

The pair is **ordered**. The literal Laura++ convention is `pair=(0, 1)`,
corresponding to `(d1, d2)`, with `theta_12` defined as the angle between `d1`
and the bachelor `d3` in the `d1 d2` rest frame. Reversing the order to
`pair=(1, 0)` leaves `m'` unchanged and reflects the angular coordinate,
`theta' -> 1 - theta'`.

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

The Laura++-compatible **default** is `gauss-legendre` with `resolution=1000`,
meaning 1000 nodes on each axis (one million two-dimensional nodes). This
matches Laura++'s Square-Dalitz prescription for treating narrow structures
along the diagonal pair. The Gauss-Legendre weights are folded into
`PhaseSpaceSample.weights` together with the physical Jacobian and scaled so
that the package-wide convention `mean(weights * f)` remains valid.

`midpoint` remains available explicitly. It uses a regular, uniform grid of cell centers in both transformed variables,

```text
m'_a     = (a + 1/2) / N
theta'_b = (b + 1/2) / N
```

so the sampling itself is uniform on `[0,1] x [0,1]`; all non-uniformity of the physical measure enters through the Jacobian.

Gauss-Legendre often converges faster for smooth functions, but narrow or highly localized amplitudes still require sufficient resolution. The Laura++ default therefore does not replace a convergence study of the normalization matrix.

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

For a `B+ -> K+ pi+ pi-` model one may use

```python
pair=(0, 2)
```

which corresponds to the `(1,3)` pair in one-based notation and therefore transforms

```text
m_13 = m(K+ pi-).
```

The Square-Dalitz sample can be used consistently for both individual-component
and total-PDF normalization.

## Validation

`tests/test_square_dalitz.py` checks:

- Laura++-compatible defaults and ordered-pair convention;
- reflection of `theta'` when the ordered pair is reversed;
- the Jacobian against Laura++'s factorized expression;
- invariant -> square-Dalitz -> invariant round trips;
- the midpoint integral of a constant against the ordinary Dalitz area;
- Gauss-Legendre constant and smooth-moment integrals;
- convergence for a narrow Breit-Wigner-like structure at sufficiently high resolution.
