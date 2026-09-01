# ROOT input with uproot

DalitzPlotFitter reads ROOT files with `uproot`; PyROOT is not required.

## TTree input

Use `read_root_tree` when arbitrary branches are needed:

```python
from dalitzplotfitter import read_root_tree

arrays = read_root_tree(
    "data.root",
    "DecayTree",
    {"s13": "B_s13", "s23": "B_s23", "mass": "B_M"},
    cut="B_M > 5200 && B_M < 5350",
)
```

The mapping is `{output_name: ROOT_branch_name}`. Returned arrays are JAX arrays. Flat scalar and fixed-size numeric branches are supported.

For a three-body fit sample use `read_phase_space_sample`:

```python
from dalitzplotfitter import read_phase_space_sample

data = read_phase_space_sample(
    "data.root", "DecayTree",
    s12="S12", s13="S13", s23="S23",
    weight="eventWeight",
)
```

If no weight branch is supplied, all event weights are one. Optional four-momenta use `(E, px, py, pz)` branch tuples for `p1`, `p2` and `p3`. `entry_start`, `entry_stop` and uproot `cut` are supported.

## ROOT histograms in ordinary Dalitz coordinates

```python
from dalitzplotfitter import histogram_background_from_root, histogram_efficiency_from_root

efficiency = histogram_efficiency_from_root(
    "maps.root", "efficiency_s13_s23",
    x_variable="s13", y_variable="s23",
)
background = histogram_background_from_root(
    "maps.root", "background_s13_s23",
    x_variable="s13", y_variable="s23",
)
```

These return `HistogramEfficiency` and `HistogramBackground`.

## ROOT histograms in Square Dalitz coordinates

For B-decay analyses it is common to store efficiency and background maps directly in Square-Dalitz coordinates `(m', theta')`. This is supported natively:

```python
from dalitzplotfitter import (
    square_dalitz_background_from_root,
    square_dalitz_efficiency_from_root,
)

kwargs = dict(
    mother_mass=channel.parent_mass,
    masses=channel.daughter_masses,
    pair=(0, 2),
)

efficiency = square_dalitz_efficiency_from_root(
    "maps.root", "efficiency_sdp", **kwargs
)
background = square_dalitz_background_from_root(
    "maps.root", "background_sdp", **kwargs
)
```

The TH2 x axis is interpreted as `m'` and the y axis as `theta'`. Usually both axes cover `[0,1]`. The ordered `pair` must be exactly the same Square-Dalitz convention used when producing the histogram.

The fitter itself continues to pass ordinary invariants `(s12,s13,s23)`. The histogram model converts every evaluation point internally with `invariants_to_square_dalitz`, then performs the TH2 bin lookup. This means the same object works transparently in:

```text
event PDF evaluation
signal normalization
toy generation
background normalization
CP fits
```

No extra Jacobian multiplies the efficiency or background histogram value. The map represents a scalar efficiency or shape evaluated at the transformed coordinates. The Square-Dalitz Jacobian belongs to coordinate integration, and is already included when a `SquareDalitzGrid` normalization sample is used.

The classes can also be constructed directly without ROOT:

```python
from dalitzplotfitter import (
    SquareDalitzHistogramEfficiency,
    SquareDalitzHistogramBackground,
)

efficiency = SquareDalitzHistogramEfficiency(
    mprime_edges,
    thetaprime_edges,
    values,
    channel.parent_mass,
    channel.daughter_masses,
    pair=(0, 2),
)
```

## Normalization

Efficiency histograms enter the signal normalization as

```text
integral epsilon(Phi) |A(Phi)|^2 dPhi.
```

A background histogram remains an unnormalized shape until its Dalitz integral is computed, for example

```python
bkg_norm = jnp.mean(norm.weights * background(norm.as_dict()))
```

For Square-Dalitz quadrature, `norm.weights` already contain the transformation Jacobian. Therefore the efficiency/background map itself must not include an additional Jacobian unless that factor was intentionally built into the external histogram definition.

## Examples

- `notebooks/13_b2kpipi_root_tree_input.ipynb`: TTree -> `PhaseSpaceSample` -> amplitude fit;
- `notebooks/14_b2kpipi_root_hist_eff_background.ipynb`: ROOT TH2 maps in ordinary Dalitz coordinates;
- `notebooks/15_b2kpipi_square_dalitz_eff_background.ipynb`: ROOT TH2 efficiency/background maps in `(m', theta')`, with SDP and ordinary-Dalitz plots and a signal/background fit.
