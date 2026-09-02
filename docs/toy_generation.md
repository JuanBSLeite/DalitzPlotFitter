# Toy generation and ROOT output

`generate_signal_toy`, `generate_toy` and `generate_cp_toy` generate unweighted pseudo-data. The default sampling method is vectorized accept-reject, with the previous fixed-pool resampling implementation available through `method="resample"`.

## Save a non-CP toy to ROOT

```python
from dalitzplotfitter import generate_toy

toy = generate_toy(
    model,
    100_000,
    parameters=truth,
    seed=1,
    output_root="toy.root",
)
```

The default tree is `DecayTree`. It contains `s12`, `s13`, `s23`, `weight`, and, when available in the generated `PhaseSpaceSample`, the four-momentum branches

```text
p1_E  p1_PX  p1_PY  p1_PZ
p2_E  p2_PX  p2_PY  p2_PZ
p3_E  p3_PX  p3_PY  p3_PZ
```

The tree name can be changed with `output_tree=`. The generated sample is still returned in memory.

## Save a CP toy to one ROOT tree

```python
plus_toy, minus_toy = generate_cp_toy(
    plus_model,
    minus_model,
    100_000,
    parameters=truth,
    seed=2,
    output_root="cp_toy.root",
)
```

Both charges are written to the same `DecayTree`. A signed integer branch named `charge` identifies the sample:

```text
charge = +1  -> B+
charge = -1  -> B-
```

This makes charge selections straightforward in uproot or ROOT while retaining one common file/tree:

```python
import uproot

with uproot.open("cp_toy.root") as f:
    tree = f["DecayTree"]
    plus = tree.arrays(cut="charge > 0", library="np")
    minus = tree.arrays(cut="charge < 0", library="np")
```

The tree and branch names are configurable:

```python
generate_cp_toy(
    plus_model,
    minus_model,
    100_000,
    output_root="cp_toy.root",
    output_tree="DecayTree",
    charge_branch="charge",
)
```

ROOT output is implemented with `uproot`; PyROOT is not required.
