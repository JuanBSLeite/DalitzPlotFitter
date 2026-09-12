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

## JAX and iminuit

`Minimizer` compiles its `jax.value_and_grad` evaluator lazily. The compiled backend is reused both inside one `Minimizer` and across short-lived `Minimizer` instances that wrap the same live objective and parameter layout. This is the common pattern produced by repeated `FitSession.fit()` or `CPFitSession.fit()` calls, so a second fit of the same session does not pay the same XLA compilation cost again.

The shared lookup stores only a weak reference to the objective. A completed fit session can therefore be garbage-collected normally instead of being retained by the compilation cache.

The Minuit value and gradient callbacks also share the last evaluated parameter point, so requesting the value and gradient at the same point causes only one JAX device evaluation and one device-to-host transfer.

`Minimizer(..., hessian="jax")` (or `session.fit(hessian="jax")`) adds a separately
compiled automatic Hessian for both MIGRAD and HESSE. It uses forward-over-reverse
AD: `jax.linearize(jax.grad(objective), point)` followed by sequential
Hessian-vector products, including through QMI's custom VJPs. It caches the last
Hessian independently of the value/gradient point. Both compiled programs are
shared across minimizers of the same live objective and fixed-parameter layout.
No Hessian program runs or compiles on the default `hessian="numerical"` path.
The JAX path supplies a diagonal callback too: iminuit 2.32's negative-curvature
recovery can call it even when a full Hessian is supplied.

The differentiation direction matters on GPU. Applying reverse mode again to
the QMI gradient reintroduces scatter-adds through the saved knot gathers,
including highly contended FP64 updates. The grouped first-derivative VJP alone
does not prevent this. Linearizing the **gradient** in forward mode preserves its
grouped reductions. An isolated prepared linear-QMI graph (49 knots, 10,000
events) contained 16 scatter operations with reverse-over-reverse and none with
the current forward-over-reverse implementation. Columns are still evaluated
sequentially to bound intermediate memory. See
[`jax.linearize`](https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html).

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
