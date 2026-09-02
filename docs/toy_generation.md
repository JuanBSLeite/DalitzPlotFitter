# Toy generation and ROOT output

`generate_signal_toy`, `generate_toy` and `generate_cp_toy` generate unweighted pseudo-data. The default sampling method is vectorized accept-reject, with the previous fixed-pool resampling implementation available through `method="resample"`.

## Current accept-reject benchmark

The accept-reject implementation follows the Laura++ safety principle: if a proposal exceeds the current envelope, already accepted events from that component are discarded and generation restarts with a larger envelope. Probabilities are never clipped.

A CPU benchmark on GitHub Actions using the full paper-inspired `B+ -> K+ pi+ pi-` model showed that the **current proposal distribution is not yet efficient for accept-reject**. The proposal is `PhaseSpaceMC`, so the rejection score contains both the phase-space proposal weight and the dynamical density. With one global envelope this gives a very low acceptance rate.

| output events | accept-reject | resample | accept-reject proposals | resample pool |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 4.06 s | 3.04 s | 2.02 M | 0.10 M |
| 100,000 | 16.80 s | 3.13 s | 12.1 M | 1.0 M |
| 1,000,000 | 96.28 s | 8.88 s | 116.1 M | 10.0 M |

The accept-reject output is fully unweighted and had no duplicated phase-space points in this benchmark. The finite-pool resampler naturally contains duplicates; its unique-`s12` fraction was about 0.80. One- and two-dimensional closure between the two methods improved with sample size and was statistically consistent.

Therefore the current performance limitation is the **proposal/envelope**, not the accept-reject principle itself. Future optimization should replace the current weighted phase-space proposal with a flatter Dalitz-space proposal or a locally/adaptively bounded proposal before changing the statistical algorithm.

The benchmark is reproducible with:

```bash
python benchmarks/benchmark_toy_generation.py --size 100000
```

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
