# Square Dalitz integration

`SquareDalitzGrid` provides deterministic integration on the square-Dalitz variables commonly used in three-body B-decay amplitude analyses.

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

a regular midpoint grid in `(m', theta')` evaluates the same physical Dalitz integral as `DalitzGrid`:

```text
integral_DP f(s12,s13,s23) ds_a ds_b
=
integral_0^1 integral_0^1 f(s(m',theta')) |J| dm' dtheta'.
```

## Usage

```python
from dalitzplotfitter import SquareDalitzGrid

normalization_sample = SquareDalitzGrid(
    model.channel.parent_mass,
    model.channel.daughter_masses,
    resolution=500,
    pair=(0, 2),
).sample()

cache = model.prepare_cache(
    data_sample,
    normalization_sample=normalization_sample,
)
```

The same sample may be supplied to `DecayModel.pdf(normalization_sample=...)`.

No amplitude, coefficient, cache, likelihood or minimizer code depends on the coordinate system used to construct the normalization sample. In particular, the normalization matrix remains

```text
M_ij = integral F_i^* F_j dPhi
```

and a coherent model is normalized as `c^dagger M c`.

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

The notebook uses the Square Dalitz grid for component normalization, the two charge integrals `I+` and `I-`, and the joint CP likelihood denominator `I+ + I-`.

## Validation

`tests/test_square_dalitz.py` checks:

- invariant -> square-Dalitz -> invariant round trips;
- the integral of a constant against the ordinary Dalitz area;
- nontrivial polynomial moments;
- an illustrative narrow Breit-Wigner-like function in `s13`.

These comparisons protect the transformation and Jacobian from silent convention errors.
