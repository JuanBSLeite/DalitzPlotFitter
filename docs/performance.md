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

If no dynamical parameter is floating, `PreparedAmplitudeCache` releases the large prepared normalization-event mapping and the per-point normalization component array after constructing the fixed normalization matrix. This can save substantial accelerator memory for high-resolution Square-Dalitz grids.

If a mass, width, radius, lineshape parameter, or other `ParameterKind.DYNAMICS` quantity floats, the relevant prepared data are retained. Only components owned by floating dynamical parameters are reevaluated.

For multiple floating dynamical components, all affected normalization-matrix rows are updated in one batched accelerator reduction rather than one full normalization-grid reduction per component.

## JAX and iminuit

`Minimizer` compiles its `jax.value_and_grad` evaluator lazily and reuses that compiled backend for the lifetime of the `Minimizer` object. The Minuit value and gradient callbacks share the last evaluated parameter point, so requesting the value and gradient at the same point causes only one JAX device evaluation and one device-to-host transfer.

The established strategy-2 refinement is intentionally retained: refined fits still run the existing two MIGRAD passes followed by HESSE. Removing the second pass changed convergence/precision in the regression suite. It should therefore only be reconsidered as an explicit fast-fit mode after dedicated closure studies.

## QMI and SCF preparation

For cubic one-dimensional QMI amplitudes, the natural-spline linear system depends only on the fixed knot coordinates. Its inverse is now cached and reused; changing magnitudes or phases no longer solves the same system from scratch.

`SquareDalitzSCFMap` caches its fixed Square-Dalitz bin centres, corresponding invariant coordinates, and phase-space areas. The dense migration matrix is still supported as before. Very fine SCF maps can remain memory intensive; a sparse migration representation is a separate future optimization because it changes storage and execution semantics.

## Benchmarking on the target GPU

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
- prepared-cache construction time;
- first value+gradient time, which includes JIT compilation;
- steady-state value+gradient timing after compilation;
- whether the amplitude cache is compact.

The first compiled call should not be confused with steady-state fit throughput. GPU/XLA compilation can be significant, while subsequent iterations are much faster.

## Precision

The project uses 64-bit real and 128-bit complex arithmetic when `enable_x64()` is enabled. This is deliberate for amplitude-analysis stability. Consumer GPUs can have much lower FP64 throughput than data-centre GPUs, but changing the default to float32/complex64 should only be done after explicit likelihood, parameter, fit-fraction, and toy-closure studies.
