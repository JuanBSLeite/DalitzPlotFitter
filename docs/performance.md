# Fit performance and GPU execution

DalitzPlotFitter separates expensive one-time preparation from repeated likelihood evaluation. This distinction is especially important on GPUs, where recomputing resonance dynamics on a large normalization grid can dominate the fit even when the final coefficient algebra is small.

## Prepared single-sample fits

`FitSession` prepares a `PreparedAmplitudeCache` before repeated likelihood calls. For fixed resonance dynamics, the cache stores the component values on the data and the normalization matrix

\[
M_{ij}=\int d\Phi\,\epsilon(\Phi) F_i^*(\Phi)F_j(\Phi).
\]

A coefficient-only likelihood evaluation then requires only

\[
A_n=\sum_i c_i F_i(x_n),\qquad
\mathcal N=c^\dagger M c,
\]

plus the event-wise log-likelihood reduction. The resonance functions are not reevaluated on the normalization grid at every Minuit step.

Efficiency and veto values on the data and normalization sample are also evaluated once by `FitSession` and reused.

## Compact coefficient-only cache

If no dynamical parameter is floating, `PreparedAmplitudeCache` uses a dedicated compact preparation path. Fixed component evaluations, component normalization and construction of the normalization matrix are compiled with JAX, while the large normalization sample is processed in fixed-size chunks.

The default normalization chunk size is 100,000 points. For a one-million-point Square-Dalitz grid, ten chunks therefore reuse the same XLA executable instead of compiling one very large graph specialized to one million points.

The matrix is accumulated as sums over chunks,

\[
M_{ij}=\frac{1}{N}\sum_k\sum_{n\in k}
 w_n F_i^*(x_n)F_j(x_n),
\]

so chunking changes only the execution schedule, not the quadrature convention. A partial final chunk is padded with a valid physical event and zero integration weights, so padded entries do not contribute.

When no efficiency map is present, the bare component matrix is needed to normalize individual components. After obtaining scales

\[
s_i = 1/\sqrt{M_{ii}},
\]

the normalized matrix is obtained algebraically as

\[
M'_{ij}=s_i M_{ij}s_j.
\]

The code therefore does not perform a second full normalization-grid reduction after scaling. With an efficiency map, the efficiency-weighted matrix is accumulated explicitly because in general it cannot be derived from the bare matrix by per-component scaling alone.

After preparation, the compact cache releases the large prepared normalization-event mapping and the per-point normalization component array. Only the data component matrix, the small fixed normalization matrix and the per-component scales remain resident for coefficient-only minimization.

The fixed model normalization is also retained by `DecayModel`. Subsequent `FitSession` objects using the same model but different datasets reuse the same normalization matrix and component scales and only evaluate the data-side amplitudes. This is particularly useful for toy, bootstrap and repeated-fit campaigns.

If a mass, width, radius, lineshape parameter, or other `ParameterKind.DYNAMICS` quantity floats, the relevant prepared data are retained. Only components owned by floating dynamical parameters are reevaluated.

For multiple floating dynamical components, all affected normalization-matrix rows are updated in one batched accelerator reduction rather than one full normalization-grid reduction per component.

For floating dynamics, `normalization_chunk_size` also bounds the prepared
normalization blocks. The requested size is a maximum: the cache balances the
effective static width below it to minimize tail padding. Each block is
accumulated with `jax.lax.scan` and checkpointed so gradients do not retain the
whole grid. Ordinary parametric lineshapes use the additional
`dynamics_microbatch_size` bound (default 20,000), which is balanced in the
same way inside each macroblock.
QMI is prepared directly in blocks no larger than that bound because its cached
sort indices cannot be sliced after preparation. The grid resolution still
controls quadrature accuracy and should not be reduced without a normalization-
convergence check.

## Input memory in multi-toy studies

`read_root_tree` defaults to JAX arrays on the active device. Keeping a complete
multi-toy file in that form consumes VRAM throughout every fit, even if each
fit only selects one toy. Use `library="np"` to read into host RAM and apply
the toy/charge mask before `jnp.asarray`; see [ROOT input](root_io.md).
Restart an existing notebook kernel to release arrays and compiled programs
from previous runs before comparing memory use.

The notebook-23 `SqDP_FreeMasses` input has 47,560,440 entries. Its three float64
invariants and two int32 labels occupy 1,521,934,080 bytes (1.417 GiB). Keeping
these arrays on the host removes that retained device payload without changing
events, precision, quadrature, free parameters, or likelihood conventions.

For this notebook's actual model, maps and complete toy 0, run:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
  python benchmarks/benchmark_pull_study_memory.py --require-gpu --hessian \
  --normalization-resolution 1000
```

This diagnostic requires the notebook's local ROOT inputs. It reads a bounded
host entry range containing toy 0, executes setup only, and reports memory after
input selection, model/map preparation, amplitude caches, value/gradient, and
optionally the automatic Hessian. It does not run the multi-toy loop or write
CSV results. `--hessian` reproduces the automatic-Hessian measurement.
`--fit-ncall N` additionally runs one fit with a call limit; a
limited run need not converge. CPU allocator statistics can be unavailable,
and JAX's live/peak allocation counters exclude some CUDA/runtime overhead.
Measure the device process as well when assessing a laptop's VRAM budget.

On the RTX 3050 Ti (4 GiB), the 2026-09-19 end-to-end check used the notebook's
95,074 accepted events, 41 free parameters, and one million normalization
points per charge. With dynamic microbatching and sequential Hessian-vector
products, the full toy-0 MIGRAD+HESSE fit completed with `hessian="jax"`,
`valid=True`, accurate covariance, EDM `9.433e-8`, and NLL
`-494077.2905459427`. The complete diagnostic, including explicit pre-fit
value/gradient and Hessian evaluations, took 219.4 s; the already-compiled fit
stage took 95.8 s. A corresponding `hessian="numerical"` run took about 1066 s
and reached the same NLL to six decimal places.

The current implementation also works with JAX's default BFC allocator. At the
notebook's 500-by-500 setting, its effective pool limit was 2.76 GiB; the
Hessian completed with a peak live JAX allocation of 438,583,552 bytes
(418.3 MiB) and a 543,162,368-byte allocator pool (518 MiB). The benchmark
finished preparation, value/gradient, and Hessian in 94.6 s. At 1000-by-1000,
the Hessian took 74.4 s and the whole diagnostic took 129.9 s. Peak live JAX
memory was 1,144,694,784 bytes (1.066 GiB), and the allocator pool reached
2,153,775,104 bytes (2.006 GiB). This check validates one complete toy, not the
full pull distribution.

## JAX and iminuit

`Minimizer` compiles its `jax.value_and_grad` evaluator lazily. The compiled backend is reused both inside one `Minimizer` and across short-lived `Minimizer` instances that wrap the same live objective and parameter layout. This is the common pattern produced by repeated `FitSession.fit()` or `CPFitSession.fit()` calls, so a second fit of the same session does not pay the same XLA compilation cost again.

The shared lookup stores only a weak reference to the objective. A completed fit session can therefore be garbage-collected normally instead of being retained by the compilation cache.

The Minuit value and gradient callbacks also share the last evaluated parameter point, so requesting the value and gradient at the same point causes only one JAX device evaluation and one device-to-host transfer.

`Minimizer(..., hessian="jax")` (or `session.fit(hessian="jax")`) adds an
automatic Hessian for both MIGRAD and HESSE. It uses forward-over-reverse AD,
including through QMI's custom VJPs. For floating dynamics,
`hessian_batch_size` controls how many Hessian-vector products one compiled
program evaluates together. Its default of 1 runs each column separately to
minimize peak memory. For coefficient-only fits, one linearization is reused
inside a single executable and this option has no effect.
The last Hessian is cached independently of the value/gradient point. Compiled
programs are shared across minimizers of the same live objective and fixed-
parameter layout. No Hessian program runs or compiles on the default
`hessian="numerical"` path. The JAX path supplies a diagonal callback too:
iminuit 2.32's negative-curvature recovery can call it even when a full Hessian
is supplied.

The differentiation direction matters on GPU. Applying reverse mode again to
the QMI gradient reintroduces scatter-adds through the saved knot gathers,
including highly contended FP64 updates. The grouped first-derivative VJP alone
does not prevent this. Applying a forward-mode JVP to the **gradient** preserves
its grouped reductions. An isolated prepared linear-QMI graph (49 knots, 10,000
events) contained 16 scatter operations with reverse-over-reverse and none with
the current forward-over-reverse implementation. Columns are still evaluated
sequentially to bound intermediate memory.

Run `python benchmarks/benchmark_hesse.py --model qmi --events 5000 --repeats 3`
to compare numerical and automatic HESSE. The benchmark reports cold and warm
times separately and changes each evaluation point to bypass the host cache.
It evaluates curvature at initial model parameters on phase-space data, **not**
at a fitted minimum; covariance flags are reported and timings do not establish
fit convergence or physics precision. Large samples still require substantial
second-order work and memory, so measure on the device used for the analysis.
With `verbose=1`, full fits report elapsed time and call counts for each optimizer
stage, including first-use JIT and any internal HESSE work within MIGRAD.

### GPU Hessian validation (2026-09-12)

The corrected backend passed all 29 `tests/test_minimizer.py` tests on a GeForce
RTX 3050 Ti Laptop GPU (4 GB), using Python 3.14, JAX 0.11.1, iminuit 2.32.0,
CUDA 13 and NVIDIA driver 595.91.07. Runs required `JAX_PLATFORMS=cuda`, enabled
x64, disabled preallocation, and explicitly checked `jax.default_backend()`;
there was no CPU fallback.

The `13_b2kkk_cpvfit_qmi` model was also checked with its 49 knots, 125 free
parameters and 1,037,441 adaptive Gauss-Legendre normalization points per charge
(nominal resolution 20, with narrow-resonance refinement). On 500/450 generated
phase-space events, the Hessian took 15.77 s on its first call, including JIT, and
2.442–2.443 s on three subsequent changed-point calls. Against central differences
of the JAX gradient (`step=1e-5`), the relative Frobenius-norm difference was
`3.03e-8`. Peak live JAX allocations were 711,382,016 bytes; the allocator pool
peaked at 1,075,838,976 bytes (these exclude some CUDA/runtime overhead).

A second run used 179,188/162,598 generated events (341,786 total), keeping the
same 125 free parameters and normalization grids. The first Hessian took 22.27 s;
three subsequent changed-point evaluations took 3.572, 3.596 and 3.596 s. The
relative Frobenius-norm difference against gradient finite differences was
`1.72e-8`. Peak live JAX allocations were 975,183,104 bytes and the allocator pool
peaked at 2,149,580,800 bytes. No out-of-memory error occurred. Timings include
device synchronization and transfer of the resulting matrix to NumPy.

Both checks retained the notebook's yield-asymmetry convention and mass/width
constraints, but used generated events without measured efficiency/background
maps. They validate derivatives and GPU execution, not the fitted data result.
The previous reverse-over-reverse implementation was interrupted before its
first GPU Hessian returned; its earlier CPU-only timing is not a measurement of
the corrected implementation.

The established strategy-2 refinement is intentionally retained: refined fits still run the existing two MIGRAD passes followed by HESSE. Removing the second pass changed convergence/precision in the regression suite. It should therefore only be reconsidered as an explicit fast-fit mode after dedicated closure studies.

### Fit-fraction uncertainties

`DecayModel.fit_fraction_errors` and the session wrappers use a compiled,
sequential reverse-mode Jacobian specialized to the model's parameter layout.
Normalization arrays, efficiencies and current parameter values are runtime
inputs, so the same executable can serve later parameter points and integration
samples without retaining their arrays. The fraction cache prepares just one
event on its unused data side; the full integration sample and weight convention
are unchanged. First-use compilation still costs time.

For CP fits, the two charges' Jacobian rows are computed separately in the same
union-of-parameter-names order, then stacked before evaluating `J @ C @ J.T`.
This avoids zero-cotangent work through the opposite charge while preserving
the cross-charge covariance needed for the mean fraction's error. The generic
`delta_method_jacobian` helper remains available with its eager row loop.

GPU comparison on the RTX 3050 Ti described above used the notebook-13 model
(125 free parameters, 16 charge/component fractions, 1,037,441 normalization
points per charge), physical fractions at the model starting values, and a
synthetic positive-definite covariance:

| Measurement | Previous eager implementation | Compiled fraction kernels |
| --- | ---: | ---: |
| First full uncertainty call, including preparation and JIT | 14.83 s | 15.06 s |
| Second full uncertainty call | 4.89 s | 1.19 s |
| Jacobian portion of the second call | 3.69 s | 0.097 s |
| Peak live JAX array allocation | 1.398 GB | 0.661 GB |

These are two calls in each of two fresh GPU processes, after constructing the
normalization grids. The benefit is primarily repeated evaluation and lower
memory use, not a faster first call. The largest absolute difference in standard
errors was `1.73e-18`; the Jacobian relative Frobenius-norm difference was
`3.00e-16`. Runtime/driver memory is additional to the JAX allocation numbers.
This comparison does not replace evaluating the actual fitted covariance and
measured efficiency maps. Tests also cover efficiency-weighted fractions,
floating QMI knots, component normalization, covariance ordering and
changed values/weights/efficiencies on a reused executable.

For a reproducible single-charge comparison, run each method in a fresh process:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/benchmark_fit_fraction_errors.py --method eager --require-gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/benchmark_fit_fraction_errors.py --method prepared --require-gpu
```

### Hazard: mismatched parameter lists between cache and `Minimizer`

`PreparedAmplitudeCache.prepare()` decides, once, which DYNAMICS parameters go
through the compact fixed-evaluation path versus the dynamic-recompute path,
based on the `fixed` flag of the `parameters` it was given. `Minimizer` is a
separate, decoupled class that accepts its own `parameters` sequence. `FitSession`/
`CPFitSession`/`DecayModel.prepare_cache` always thread the same `model.parameters`
into both, so this cannot drift in the documented high-level workflow. But an
advanced caller assembling `PreparedAmplitudeCache` and `Minimizer` directly (see
`docs/user_friendly_api.md`) must pass the *same* parameter list to both: handing
`Minimizer` a list that marks a DYNAMICS parameter as floating when the cache was
prepared with it fixed produces no error, and Minuit sees an exact zero gradient
along that direction instead of a small one — the value simply never reaches the
cache's compact evaluation path. Call `cache.check_parameters(parameters)` before
constructing `Minimizer` whenever the two parameter lists are not obviously the
same object.

### AD microbatching for floating-dynamics normalization on constrained GPUs

A follow-up review fixed the compatibility helper `_matrix_from_dynamic` for the
chunked cache representation, corrected retained-memory accounting, and made
the shared Hessian callback capture its configured batch size. It also replaced
an invalid test assertion about private backend-tuple identity with checks that
the compiled callbacks are actually shared; see
[the 2026-09-19 review](reviews/20260919_dynamics_chunking_and_hessian_review.md).

`normalization_chunk_size` (default 100,000) is chosen to amortize XLA
compilation and geometry-storage cost, not to bound the memory of the reverse-AD
pass through several floating `DYNAMICS` lineshapes at once. On a memory-constrained
GPU, differentiating a single 100,000-point macro-chunk's worth of parametric
lineshapes (e.g. several `GounarisSakurai`/`RelativisticBreitWigner`/`SigmaPole`
components with barrier and angular factors) can itself exceed available memory,
independent of how many macro-chunks the sample is split into — a 4 GB laptop
GPU (RTX 3050 Ti) fitting a real `B -> 3pi` CP model (41 free parameters, 4
floating-mass/width resonances, `normalization_resolution=1000` i.e.
1,000,000-point Square-Dalitz grids per charge) hit `RESOURCE_EXHAUSTED` trying
to allocate 1.4 GiB on the very first MIGRAD gradient call, well before ever
reaching HESSE.

`PreparedAmplitudeCache._chunked_dynamic_normalization` therefore re-splits each
macro-chunk into `DecayModel(..., dynamics_microbatch_size=20_000)` microbatches
via a second, nested
`jax.lax.scan` wrapped in its own `jax.checkpoint`, so the reverse-AD pass never
has to hold more than one microbatch's forward residuals live at a time —
independent of `normalization_chunk_size` or the total grid size. The shown
value is the default. This fixed
the 1.4 GiB allocation above outright: the same model/data/GPU then ran a full
MIGRAD+HESSE (`hessian="numerical"`) at `normalization_resolution=1000` to
completion (`valid=True`, EDM `9.4e-8`), taking about 1066 s per toy — roughly
proportional to the ~11x more normalization points than the 90,000-point grid
(`normalization_resolution=300`) that already fit unchunked, confirming the added
nested-scan/checkpoint machinery costs kernel-launch overhead, not a multiplicative
blowup in wall time.

Larger values reduce scan overhead and can be faster when the GPU has enough
memory. Smaller values reduce peak gradient and Hessian memory. Changing this
option changes only evaluation partitioning, not the normalization integral or
its quadrature resolution.

For ordinary floating-dynamics lineshapes, `dynamics_microbatch_parallelism`
evaluates several microbatches concurrently with `jax.vmap` and reduces their
partial normalization blocks. Its default is 1, preserving the sequential
bounded-memory path. Values of 2 or 4 can improve throughput on larger GPUs,
but increase peak AD memory approximately with the number of concurrent
microbatches. The effective value is reported by
`cache.effective_dynamics_microbatch_parallelism`; it is capped by the number
of microbatches in one macro-chunk. QMI keeps the effective value at 1 because
its prepared sort order is local to the complete block.

For a floating component whose lineshape sets
`prepared_mass_is_order_dependent = True` (currently only `QMI`), preparation
itself uses blocks no larger than `dynamics_microbatch_size`. `QMI.prepare_mass`
sorts one prepared block by knot interval and returns `order`/`starts`/`ends`
indices valid only for that exact block, so reshaping a larger prepared block
would desynchronize those indices from the custom VJP. `KMatrix.prepare_mass`
stores pointwise per-event responses and uses the ordinary inner microbatch
path.

### Bounded-memory `hessian="jax"` for floating dynamics

The automatic Hessian originally put all columns into one XLA executable:
`jax.linearize(jax.grad(vector_objective), point)` produced one pushforward,
then `jax.lax.map` evaluated the basis vectors. Checkpointing the gradient
reduced its program estimate substantially, but the real notebook model still
required about 3.30 GiB at 500-by-500 resolution. That exceeded the default
BFC allocator's 2.76 GiB pool on the 4 GiB RTX 3050 Ti.

For floating `DYNAMICS` parameters, `Minimizer` now checkpoints the gradient
and compiles a single forward-over-reverse Hessian-vector product:

```python
gradient = jax.checkpoint(jax.grad(vector_objective))
hvp = jax.jit(lambda point, tangent: jax.jvp(
    gradient, (point,), (tangent,),
)[1])
```

With the default `hessian_batch_size=1`, the host invokes this reusable program
once per basis vector and transfers each column before starting the next.
Temporary buffers from different columns therefore cannot overlap, and XLA
compiles one HVP rather than a program that contains the entire Hessian loop.
The differentiation remains
forward-over-reverse, preserving QMI's grouped custom-VJP reductions. The
result is symmetrized once after the columns are assembled, as before.

On a larger GPU, `hessian_batch_size=2`, `4`, or `8` evaluates that many HVPs
with `jax.vmap` in each device execution. This trades temporary memory for fewer
launches and more parallel work. For example:

```python
model = DecayModel(
    channel,
    components,
    dynamics_microbatch_size=100_000,
)
session = FitSession(model, data)
result = session.fit(
    hessian="jax",
    hessian_batch_size=4,
    strategy=1,
)
```

Tune the two options independently: increase `dynamics_microbatch_size` first
for the repeatedly evaluated likelihood/gradient, then benchmark
`hessian_batch_size` for automatic-Hessian calls. If either setting exhausts
VRAM, reduce it. The defaults (`20_000` and `1`) retain the validated 4 GiB
behavior.

To compare macro- and microbatch combinations on a target GPU, run:

```bash
python benchmarks/benchmark_dynamics_chunking_sweep.py \
  --events 100000 --normalization-resolution 500 \
  --chunk-sizes 50000,100000,200000 \
  --microbatch-sizes 20000,25000,40000,50000,100000
```

Both size options are upper bounds. For `N=250000`, for example, a requested
macro limit of 200000 becomes two effective blocks of 125000 instead of two
fixed blocks totaling 400000 positions; an inner limit of 100000 then becomes
62500, giving exact division at both levels. When exact division is impossible
with one static XLA shape, the balanced layout leaves fewer padded positions
than the number of blocks rather than a large partial tail. Inspect
`cache.effective_normalization_chunk_size`,
`cache.effective_dynamics_microbatch_size`,
`cache.normalization_padding_points`, and
`cache.normalization_padding_fraction` after preparation. The sweep benchmark
reports the requested and effective sizes plus the residual fraction. Compare
steady objective time as well as compilation and preparation time; retained
cache size is controlled mainly by the macro chunk representation and need not
fall with a smaller microbatch.

Coefficient-only fits keep the previous single-linearization program. They do
not need the extra memory boundary, and a small eight-parameter GPU benchmark
showed that applying the checkpoint there increased the warm Hessian time from
about 18 ms to 22 ms.

On the real `B -> 3pi` model at 500-by-500 resolution, the sequential-HVP
implementation completed with the default allocator in 46.2 s for the Hessian.
Peak live JAX memory was 418.3 MiB and the allocator pool reached 518 MiB,
compared with the previous 3.30 GiB program requirement. The computed Hessian
norm was identical to the platform-allocator run (`4892002.165911924`).

At 1000-by-1000 resolution, also with the default allocator, the Hessian took
74.4 s. Peak live JAX memory was 1.066 GiB and the pool reached 2.006 GiB,
below its 2.76 GiB limit. A contemporaneous `nvidia-smi` sample showed
3,524 MiB including CUDA/runtime memory outside JAX's counters.

The end-to-end default-allocator check at one million normalization points per
charge completed MIGRAD+HESSE with `valid=True`, accurate covariance, EDM
`9.433e-8`, and NLL matching the numerical-HESSE run to six decimal places.
MIGRAD used 39 function calls; the numerical version took about 1066 s and
needed 2411. A live notebook should still avoid retaining unrelated device
arrays, but the platform allocator is no longer part of the required
configuration.

## QMI preparation

For the local one-dimensional QMI modes (`linear`, `cubic`, `hermite`), `prepare_mass` caches the
fixed knot interval index and interpolation fraction (plus the event order and per-interval
`starts`/`ends` used by the reverse pass) once; each mode's forward evaluation and its
hand-written `jax.custom_vjp` then cost one gather and one grouped-interval-sum reduction per
event, with no global linear solve and no event-sized reverse scatter-add. `natural` (the global
natural cubic spline) is different: it is *not* on this fixed/cached path. Every evaluation
solves its knot-sized tridiagonal system from scratch with `jnp.linalg.solve`, through ordinary
JAX autodiff rather than a custom VJP, so changing magnitudes or phases re-solves the system;
benchmark it separately from the local modes for large fits (see `docs/lineshapes.md`).

For `QMI2D`, fixed interpolation geometry is also cached: bin edges, bin centres, active masks and the nearest-active gather map used to fill ghost cells. Floating magnitudes/phases therefore update only the value field and interpolation algebra, not the geometry construction.

## K-matrix preparation

The five-channel K-matrix has a particularly expensive fixed operation,

\[
D(s)=\left[I-iK(s)\rho(s)\right]^{-1}.
\]

For the pi-pi production amplitude only the first row of `D(s)` is needed. During prepared resonance evaluation, DalitzPlotFitter now stores that row once for each event/normalization point. A later change of production coefficients therefore evaluates

\[
F_{\pi\pi}(s)=D_{0j}(s)P_j(s)
\]

as a five-term complex contraction rather than repeating a 5x5 linear solve for every point at every likelihood evaluation.

Only five complex values per point are retained, rather than the full 5x5 inverse matrix. This keeps the GPU memory cost substantially lower while removing the dominant repeated K-matrix linear algebra. The ordinary standalone `KMatrix.amplitude_vector()` API still performs the full solve because it returns all five output channels.

## Sparse SCF migration

`SquareDalitzSCFMap` caches its fixed Square-Dalitz bin centres, invariant coordinates and phase-space areas. It also supports sparse migration storage.

A dense migration matrix has memory complexity

\[
O(N_{\rm bin}^2),
\]

which becomes costly very quickly. A `100 x 100` Square-Dalitz map has 10,000 bins and therefore 100 million migration elements, or roughly 800 MB in float64.

For local SCF migration, most of those elements are zero. `SparseMigration` stores only

```text
true bin index
reconstructed bin index
probability
```

for non-zero transitions, so memory becomes `O(nnz)`. Migration is evaluated with JAX gather/scatter operations and remains JIT-compatible and differentiable.

Dense input remains supported. `SquareDalitzSCFMap(storage="auto")`, the default, compresses a dense matrix when its non-zero fraction is at most 25%. For very large maps, construct `SparseMigration` directly so the dense matrix never exists.

A dedicated benchmark compares dense and sparse execution on the active JAX device:

```bash
python benchmarks/benchmark_scf_migration.py \
  --bins-mprime 40 \
  --bins-thetaprime 40 \
  --neighbors 9 \
  --repeats 50
```

The output reports storage reduction, first JIT call, steady-state execution time and the sparse/dense speed ratio. The best representation can be device- and sparsity-dependent; sparse storage is primarily essential for controlling memory at fine binning.

## Benchmarking cache compilation stages

Use the dedicated cache-stage benchmark to separate XLA lowering, compilation and execution:

```bash
python benchmarks/benchmark_cache_stages.py \
  --events 100000 \
  --normalization-resolution 1000 \
  --repeats 5
```

For the realistic five-component B+ -> K+ pi+ pi- model, one CUDA benchmark with one million normalization points showed the original full-grid normalization graph spending about 15.2 s in XLA compilation while the actual one-million-point execution took only about 0.25 s. This identified compilation, not arithmetic throughput, as the dominant cold-start cost.

With 100,000-point normalization chunks, the same device measured approximately:

- 3.18 s to compile the normalization chunk kernel;
- 0.034 s for the first 100,000-point chunk execution;
- 0.029 s per warm chunk;
- about 0.43 s for a warm traversal of all ten chunks;
- 1.30 s to compile the separate 100,000-event data kernel;
- about 0.026 s for a warm 100,000-event data evaluation.

These numbers are hardware- and load-dependent; they are provided as a representative diagnostic, not a guaranteed performance target.

## Benchmarking the full fit on the target GPU

Use the realistic five-component B+ -> K+ pi+ pi- benchmark:

```bash
python benchmarks/benchmark_fit_evaluation.py \
  --events 100000 \
  --normalization-resolution 1000 \
  --repeats 20
```

The JSON output reports:

- JAX backend and device;
- whether x64 is enabled;
- number of data and normalization points;
- normalization-grid construction time;
- phase-space data generation time;
- `prepared_cache_seconds`, including first compact-cache compilation and execution;
- `prepared_cache_reuse_seconds`, the cost of accessing the already prepared session cache;
- preparation times for second and third datasets sharing the same `DecayModel` normalization;
- first value+gradient time, which includes likelihood JIT compilation;
- steady-state value+gradient timing after compilation;
- whether the amplitude cache is compact;
- minimum and maximum normalization-matrix diagonal values as a quick component-normalization sanity check.

On the same representative CUDA setup with 100,000 data events and one million normalization points, the coefficient-only cache preparation improved from an initial baseline of about 28.95 s to about 7.24 s after compact-cache fusion, elimination of the redundant matrix reduction, model-level normalization reuse and chunked normalization compilation. The steady-state value+gradient time remained about 4.8 ms, and the normalized matrix diagonal stayed at unity to floating-point precision.

The first compiled call should not be confused with steady-state fit throughput. GPU/XLA compilation can be significant, while subsequent iterations are much faster.

## Precision

The project uses 64-bit real and 128-bit complex arithmetic when `enable_x64()` is enabled. This is deliberate for amplitude-analysis stability. Consumer GPUs can have much lower FP64 throughput than data-centre GPUs, but changing the default to float32/complex64 should only be done after explicit likelihood, parameter, fit-fraction, and toy-closure studies.
