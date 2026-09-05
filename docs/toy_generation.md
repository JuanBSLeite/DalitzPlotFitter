# Toy generation and ROOT output

`generate_signal_toy`, `generate_toy` and `generate_cp_toy` generate unweighted pseudo-data. There are exactly two public sampling methods:

```text
inverse-transform
accept-reject
```

`inverse-transform` is the default because it is dramatically faster for the full amplitude model while retaining continuous, duplicate-free events. `accept-reject` remains available explicitly as an independent reference and validation method.

## Default: inverse transform on the Dalitz plane

The simplest call now uses inverse transform automatically:

```python
toy = generate_toy(
    model,
    100_000,
    parameters=truth,
    seed=2,
)
```

This is equivalent to

```python
toy = generate_toy(
    model,
    100_000,
    parameters=truth,
    method="inverse-transform",
    inverse_resolution=1024,
    seed=2,
)
```

For a three-body decay the target density in the conventional mass-squared Dalitz plane is, up to the common phase-space constant,

\[
p(s_{12},s_{13}) \propto \epsilon(s_{12},s_{13})\,V(s_{12},s_{13})\,|\mathcal A(s_{12},s_{13})|^2.
\]

The implementation uses a numerical Rosenblatt transform. The physical Dalitz region is parameterized by

\[
s_{13}=s_{13}^{\min}(s_{12})+v\left[s_{13}^{\max}(s_{12})-s_{13}^{\min}(s_{12})\right],\qquad 0\le v\le1,
\]

and the tabulation grid is uniform in \(m_{12}=\sqrt{s_{12}}\), which resolves narrow resonances better than a grid uniform directly in \(s_{12}\). The transformed marginal contains the exact Jacobian

\[
2m_{12}\left[s_{13}^{\max}(s_{12})-s_{13}^{\min}(s_{12})\right].
\]

The sampler first inverts the marginal CDF in \(m_{12}\), then the conditional CDF in \(v\). No event is rejected and the generated invariants are continuous rather than selected from a finite candidate pool.

Four-momenta are reconstructed in the parent rest frame from the sampled invariants and then given a random global orientation, so ROOT toy output retains the same momentum branches as accept-reject generation.

`inverse_resolution` controls the number of grid points in both Dalitz directions. `inverse_quantile_resolution` can be supplied separately for the tabulated conditional inverse CDF; by default it equals `inverse_resolution`.

## Reuse the prepared inverse CDFs

The expensive part of inverse-transform generation is preparation. For repeated pseudoexperiments with the same model parameters, prepare once:

```python
from dalitzplotfitter import prepare_inverse_toy_generator

prepared = prepare_inverse_toy_generator(
    model,
    parameters=truth,
    efficiency=efficiency,
    veto=veto,
    resolution=1024,
)

toy1 = prepared.generate(100_000, seed=1)
toy2 = prepared.generate(100_000, seed=2)
toy3 = prepared.generate(1_000_000, seed=3)
```

The amplitude, efficiency and veto are evaluated on the CDF grid only during preparation. Subsequent `generate` calls perform random-number generation, CDF inversion, four-momentum reconstruction and optional component shuffling only.

If model parameters change, the prepared generator must be rebuilt because the target CDFs change.

## Accept-reject reference sampler

```python
toy = generate_toy(
    model,
    100_000,
    parameters=truth,
    method="accept-reject",
    seed=1,
)
```

The accept-reject implementation follows the Laura++ safety principle: if a proposal exceeds the current envelope, already accepted events from that component are discarded and generation restarts with a larger envelope. Probabilities are never clipped.

The current accept-reject proposal is `PhaseSpaceMC`. It is intentionally retained as an independent validation path, but its single global envelope is inefficient for strongly structured amplitude models.

## Performance

On GitHub Actions CPU with the full paper-inspired `B+ -> K+ pi+ pi-` model and `inverse_resolution=1024`, the benchmark gave:

| Events | Accept-reject | Inverse total | Inverse prepared |
|---:|---:|---:|---:|
| 10,000 | 3.49 s | 1.63 s | 0.0235 s |
| 100,000 | 15.74 s | 1.84 s | 0.115 s |
| 1,000,000 | 93.01 s | 3.28 s | 1.245 s |

For one million events, inverse transform was about 28x faster including CDF preparation and about 75x faster once the sampler had already been prepared. The prepared rate was approximately `8.0e5 events/s` on CPU.

## Accuracy and validation

Inverse-transform sampling is interpolated rather than analytically exact. Its numerical accuracy is controlled by the preparation resolution and should be validated against deterministic projections and the accept-reject reference sampler.

The repository benchmark compares the two methods using the full B+ -> K+ pi+ pi- model:

```bash
python benchmarks/benchmark_toy_generation.py --size 100000
```

The benchmark reports:

- accept-reject end-to-end time and proposal efficiency;
- inverse-transform preparation + first-generation time;
- generation time from an already prepared inverse sampler;
- 1D projection closure and 2D Dalitz total-variation distance.

The benchmark workflow is manual (`workflow_dispatch`) because the 1,000,000-event jobs are intentionally expensive.

## Method-specific options

`pool_size`, `batch_size`, `envelope_safety` and `max_restarts` belong to `accept-reject`.

`inverse_resolution` and `inverse_quantile_resolution` belong to `inverse-transform`.

Passing `pool_size` or `batch_size` together with the default inverse-transform method is rejected rather than silently ignored. If those options are needed, set `method="accept-reject"` explicitly.

## Compact toys for memory-constrained fits

By default generated toys retain reconstructed four-momenta together with the
three Dalitz invariants. If the downstream fit, plotting, efficiency and veto
models use only `s12`, `s13` and `s23`, the four-momenta can be omitted:

```python
toy = generate_toy(
    model,
    1_000_000,
    parameters=truth,
    seed=2,
    include_momenta=False,
)
```

For float64 arrays this reduces the retained array payload for one million
unweighted events from about 128 MiB to about 32 MiB. In the inverse-transform
path the momentum reconstruction is skipped entirely, so the peak memory and
generation work are also reduced. The Dalitz invariants and event weights are
identical to a generation with the same seed and `include_momenta=True`.

Existing samples can be compacted with

```python
compact = sample.without_momenta()
print(compact.nbytes)
```

The default remains `include_momenta=True` for backward compatibility and for
workflows that need momentum branches in ROOT output.

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

The signal and background charge splits use the same deterministic accepted-integral convention in both sampling paths. Both charges are written to the same `DecayTree`. A signed integer branch named `charge` identifies the sample:

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

ROOT output is implemented with `uproot`; PyROOT is not required.
