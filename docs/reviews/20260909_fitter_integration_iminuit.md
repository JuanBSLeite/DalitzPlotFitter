# Fitter review: integration, JAX and iminuit — 2026-09-09

Historical review of the single-sample FitSession path, amplitude caches, deterministic integration and Minimizer. Tests ran in separate CPU processes limited to two cores, without using the GPU running Genfit 03. The original review did not cover CP, backgrounds, SCF or every lineshape. Applied corrections are listed below.

## Confirmed findings

### 1. High: shared backend reused stale free-parameter settings

The backend signature in `fit/minimizer.py` included names, fixed flags and fixed values, but retained the first Minimizer's free-parameter tuple. Reusing the same objective with different starts, steps or bounds could configure Minuit with the old declarations.

Reproduction: `(x-3)**2`, first with `Parameter('x',0,bounds=(-5,5))`, then with `Parameter('x',0.5,bounds=(0,1))`. Preparing the first backend before fitting the second produced `valid=True`, bounds `(-5,5)` and x=2.999811927892423. The reused default was 0 instead of 0.5.

This requires reuse of the same objective instance. It does not by itself explain secondary minima in Genfit 01 or contamination between distinct objectives in Genfit 03.

### 2. Medium: FitSession prevented fixed-dynamics normalization reuse

FitSession always passed an acceptance vector, including all-ones acceptance, while DecayModel required `efficiency_normalization is None` for template reuse. With a constant model, a resolution-10 Square-Dalitz grid and two ten-event sessions, `_fixed_normalization_templates` remained empty. Calling `model.prepare_cache(data)` directly created a template.

This caused repeated work and allocations, without demonstrated numerical bias. Floating dynamics must still be reevaluated.

### 3. Execution limits: ncall is not a total fit budget

SIMPLEX and HESSE originally received no ncall; strategy 2 called MIGRAD twice, each with the requested budget. For a quadratic, `fit(ncall=1,simplex=True,hesse=True)` used 37 evaluations at strategy 1 and 51 at strategy 2. Entire stages were unrestricted, beyond ordinary approximate-limit overshoot. See the [iminuit reference](https://scikit-hep.org/iminuit/reference.html#iminuit.Minuit.migrad).

## Integration and statistical limits

- `mean(weights*f)`, the grid Jacobian `4*m13*m23`, retained-point weight scaling and the conjugation in `c†Mc` were consistent.
- Floating components and dynamic–fixed/dynamic–dynamic blocks are reevaluated; JAX gradients include those normalization dependencies.
- Minuit convergence does not certify quadrature precision. Narrow-resonance refinement is fixed from nominal parameters. Moving poles or narrowing widths may leave the well-resolved region. Inspection alone did not establish bias in Genfit 03.
- Asimov checks on the same integration support test algebraic consistency, not error relative to the continuous integral. Refine the grid and compare objectives, gradients and fitted parameters.
- The reviewed dynamic path evaluated components on the full normalization sample and retained fixed values for cross terms. Session cleanup reduces retention between fits, not the peak memory within one fit. Differentiable chunked dynamic integration remains a separate structural improvement.
- `-sum(log p)`, `errordef=0.5`, explicit parameter ordering and `jax.value_and_grad` were consistent. `valid=True` indicates local convergence; HESSE provides local errors, not a global-minimum certificate.

## Applied corrections and validation

Compiled callbacks remain shared, but free declarations now come from the current Minimizer. Regressions cover bounds, steps, defaults and explicit starts. FitSession passes None when efficiency and veto are absent. ncall is forwarded to SIMPLEX and HESSE and documented as an approximate per-stage limit; no global budget was introduced.

Initial validation: 26 tests in 28.68 s (`test_minimizer`, `test_integration`, `test_gauss_legendre_integration`, `test_dynamic_fit_consistency`, `test_model_normalization_reuse`), plus 20 tests in 10.99 s (`test_workflow`, `test_amplitude_cache`): **46 passed**. Separate reproductions established the findings.

After corrections: **53 distinct tests passed** across those seven files. Four new tests initially used a mock without fval; that test setup was corrected to wrap real iminuit methods. The minimizer file then passed 16/16; the other 37 had passed. `git diff --check` passed. Whole-file lint still reported pre-existing style issues. These changes did not implement dynamic chunking or adaptive grids during minimization.
