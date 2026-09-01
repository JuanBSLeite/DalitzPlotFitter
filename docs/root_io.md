# ROOT input with uproot

DalitzPlotFitter reads ROOT files with `uproot`; PyROOT is not required.

## TTree input

Use `read_root_tree` when arbitrary branches are needed:

```python
from dalitzplotfitter import read_root_tree

arrays = read_root_tree(
    "data.root",
    "DecayTree",
    {
        "s13": "B_s13",
        "s23": "B_s23",
        "mass": "B_M",
    },
    cut="B_M > 5200 && B_M < 5350",
)
```

The mapping is `{output_name: ROOT_branch_name}`. Returned arrays are JAX arrays.
Flat scalar and fixed-size numeric branches are supported. Jagged/object-valued branches are rejected because the amplitude-fit event representation is rectangular.

For a three-body fit sample use `read_phase_space_sample`:

```python
from dalitzplotfitter import read_phase_space_sample

data = read_phase_space_sample(
    "data.root",
    "DecayTree",
    s12="S12",
    s13="S13",
    s23="S23",
    weight="eventWeight",  # optional
)
```

If no weight branch is supplied, all event weights are set to one.

Optional four-momenta can be read with

```python
data = read_phase_space_sample(
    "data.root",
    "DecayTree",
    s12="S12",
    s13="S13",
    s23="S23",
    p1=("p1_E", "p1_PX", "p1_PY", "p1_PZ"),
    p2=("p2_E", "p2_PX", "p2_PY", "p2_PZ"),
    p3=("p3_E", "p3_PX", "p3_PY", "p3_PZ"),
)
```

The ordering is always `(E, px, py, pz)`.

`entry_start` and `entry_stop` can be used for partial reads. `cut` is forwarded to uproot's expression filter.

## ROOT histogram input

A ROOT TH2-like object can be read with

```python
values, x_edges, y_edges = read_root_histogram2d(
    "maps.root",
    "efficiency_s13_s23",
)
```

More commonly, construct fitter objects directly:

```python
from dalitzplotfitter import (
    histogram_background_from_root,
    histogram_efficiency_from_root,
)

efficiency = histogram_efficiency_from_root(
    "maps.root",
    "efficiency_s13_s23",
    x_variable="s13",
    y_variable="s23",
)

background = histogram_background_from_root(
    "maps.root",
    "background_s13_s23",
    x_variable="s13",
    y_variable="s23",
)
```

These return the existing `HistogramEfficiency` and `HistogramBackground` classes, so ROOT histograms use exactly the same evaluation and normalization path as histograms constructed from NumPy/JAX arrays.

The ROOT histogram axes must match the declared Dalitz variables. For example, a histogram stored as `(s13, s23)` must be loaded with `x_variable="s13"` and `y_variable="s23"`.

## Normalization

Efficiency histograms enter the signal normalization as

```text
integral epsilon(Phi) |A(Phi)|^2 dPhi.
```

A background histogram remains an unnormalized shape until its Dalitz integral is computed with the model normalization sample, e.g.

```python
bkg_norm = jnp.mean(
    norm.weights * background(norm.as_dict())
)
```

The same convention is used for functional and ROOT-loaded background models.

## Examples

- `notebooks/13_b2kpipi_root_tree_input.ipynb`: TTree -> `PhaseSpaceSample` -> amplitude fit;
- `notebooks/14_b2kpipi_root_hist_eff_background.ipynb`: ROOT TH2 efficiency/background maps, plots and a signal/background fit.
