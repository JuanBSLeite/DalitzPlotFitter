# Monte Carlo integration review — 2026-09-10

Scope: PhaseSpaceMC, external `normalization_sample` inputs, GridIntegrator, the normalization matrix, efficiency/veto, and conventions between MC and quadrature. The findings describe the initial review; applied fixes are recorded at the end.

## Main result

The usual weighted-PhaseSpaceMC path conserves the measure implemented by the generator. The normalization matrix agrees with direct integration. The problems found lie in the validation of external samples and in usage contracts that allow proposals, signal toys, and integration samples to be confused with one another.

## 1. High: documentation lets a signal toy be interpreted as an integration sample with unit weights

`src/dalitzplotfitter/decay.py`, `normalization_sample` documentation and constructor; `tests/test_decay_model.py::test_unweighted_toy_mc_sample_uses_unit_weights`.

The API uses the supplied weights directly. The phrase "an unweighted toy uses unit weights" does not distinguish uniform phase space from a signal toy. For points distributed according to q, the mean of f with unit weights estimates the expectation of f under q; integrating in the desired measure requires importance weights proportional to 1/q. Unit weights are appropriate only when the proposal is uniform in that measure, with the corresponding volume convention.

Reproduction in D+ -> pi- pi+ pi+: 50000 inverse-transform events with density proportional to s12, seed=1, resolution=128. Control: 200000 PhaseSpaceMC events, seed=2.

| Estimator of the phase-space mean of s12 | Result (GeV²) |
|---|---:|
| Phase-space MC, with its own weights | 1.185508 |
| Signal toy, unit weights | 1.623819 |
| Same toy, corrected proportional to 1/s12 | 1.182811 |

The existing test only verifies that the weights are used, not that the generation distribution matches the integral's measure. There is no algebraic error in GridIntegrator in this case; a clear contract for the external sample's proposal is missing. The same caution applies to MC already selected by efficiency: multiplying by efficiency again can double-count it.

Recommendation: document and, where possible, explicitly represent the proposal/measure. Offer an importance-weight path and distinguish pseudo-data generation from normalization generation. Do not infer "uniform" from unit weights alone.

## 2. Medium: the public stratified sample carries proposal weights, not full integration weights

`src/dalitzplotfitter/kinematics/phase_space_mc.py`, `_generate_three_body_invariant_cells` and `generate_stratified_invariants`.

Cells are chosen with non-uniform probabilities, but the returned weight does not include the inverse of that probability. This is intentional for accept-reject, whose envelope/cell probability supplies the correction. However, the returned object is an ordinary PhaseSpaceSample and can be passed directly to integration.

Reproduction: parent mass 2, massless daughters, 200000 events, seed=12, two cells in s12 with probabilities [0.9,0.1]. In the ds12 ds13 measure, whose exact area is 8:

- Mean of the returned weights, converted to area: **11.221280**.
- Including the `1/(n_cells*p_cell)` correction: **7.983149**.

Recommendation: explicitly distinguish proposals from integrable samples, or offer a corrected-weight option. Do not silently change these weights: accept-reject depends on the current convention. Zero probabilities also need attention, since they make regions omitted by reweighting unrecoverable.

## 3. Medium: filtering a normalization sample changes the estimator's denominator

`src/dalitzplotfitter/veto.py::VetoMap.apply`, `kinematics/sample.py::PhaseSpaceSample.take`, `integration/grid.py::GridIntegrator.integrate`.

`apply` removes events and preserves their weights. The subsequent integration divides by the retained count, not the original count. This is appropriate as a data selection, but it does not preserve the original sample's integral. The current workflow avoids this problem by multiplying the veto into the full sample; the risk appears when the user supplies an already-filtered external sample.

Reproduction in the massless case above, veto accepting s12<2 (exact area 6), seed=12:

- Mean of weight × veto on the full sample: **5.974845**.
- Mean of weights after filtering: **11.985647**, with 99700 of 200000 events retained.

Recommendation: keep events with zero contribution, or, when filtering an integration sample, multiply the retained weights by N_retained/N_original. An integration-filtering API should preserve this information. Common scales may not change the shape of an isolated fit, but absolute integrals and rate comparisons across samples need the correct convention.

## 4. Medium: non-finite external samples enter normalization

`src/dalitzplotfitter/decay.py`, `normalization_sample` validation in the constructor.

The constructor checks type, size, and shape of the vectors, but not finiteness. A physical 100-event sample with just one weight replaced by NaN was accepted; the model intensity started returning NaN because of the component normalization scale.

Recommendation: validate finite invariants and weights on input, reject samples with no usable contribution, and make the policy on negative weights explicit. Validation should also cover alternative samples supplied directly to `pdf`/`prepare_cache`, not only the constructor.

## Approved conventions and controls

- PhaseSpaceMC returns weights in the implemented physical convention `Delta(s12)*width/(128*pi³*M²)`. Quadratures use ds12 ds13. To compare their absolute values, the tests multiplied the MC weights by `128*pi³*M²`. Do not confuse this measure constant with sampling bias.
- Massless analytic case: MC area **7.981377 ± 0.010312** (estimated standard error), for exact area **8**; 200000 events, seed=12.
- In D+ -> pi- pi+ pi+, integrating `(1+0.2*s12)*(0.2+0.1*s12)*(s13>0.5)`: MC **1.520568 ± 0.002347**, Square-Dalitz quadrature at resolution 128 **1.524523**. The grid also has discretization error, especially at the veto discontinuity; the comparison is compatible and does not prove precision for narrow resonances.
- For two complex components, complex coefficients, and non-constant acceptance, the matrix and the direct sum on the same sample differed relatively by **7.74e-16**.
- The external sample is reused by the model/cache: the fit does not need to redraw it at every evaluation. This keeps the NLL deterministic for that sample; it does not remove MC integration uncertainty.
- `docs/fitting.md` and `docs/user_friendly_api.md` still describe normalization as exclusively deterministic, contradicting the current support for `toy-mc`.

## Validation and limits

**39 tests passed in 14.48 s**: `test_integration.py`, `test_phasespace_mc.py`, `test_model_normalization_reuse.py`, and `test_decay_model.py`, on CPU, two cores, float64. Additional numerical reproductions described above.

This review did not include a pull campaign or a comparison against narrow resonances across multiple seeds. To validate a fit with MC normalization, vary the integration sample's size and seeds, keeping each sample fixed during its fit, and compare against converged quadrature. Integration error is not automatically included in the iminuit covariance.


## Applied fixes

- `PhaseSpaceSample.with_importance_weights(q)` replaces weights with 1/q, validating shape, finiteness, and positivity. The contract requires the user to supply the correct proposal and clarifies the support/normalization limits; the generation distribution cannot be deduced from the points.
- The stratified generators of PhaseSpaceMC and DecayModel use `integration_weights=True` by default. They normalize the probabilities and include 1/(n_cells*p_cell); integration requires positive probability in every cell. Accept-reject explicitly requests `integration_weights=False`, preserving its envelope algorithm.
- `select_for_integration(mask)` and `veto.apply(sample, for_integration=True)` scale the weights by N_retained/N_original. Empty selections or ones with no positive weight are rejected. `take` and the default veto application remain data selections.
- External samples are validated in the DecayModel constructor and at the explicit `pdf` and `prepare_cache` entry points: finite vectors, consistent sizes, non-negative weights with a positive contribution, and, when present, complete and finite four-momenta. The policy does not allow negative subtraction weights for normalization.
- `docs/mc_integration.md` presents the contracts and examples; the old descriptions of exclusively deterministic normalization were corrected. The measure difference between physical MC weights and quadrature remains explicit.

Validation: **89 passed, 1 deselected in 48.14 s**, covering integration, phase space, DecayModel, normalization reuse, and toys. Plus **4 passed in 3.23 s** covering veto and an additional importance-weight control for q(x)=2x, recovering the integral of x as 1/2. Total: **93 tests passed** on CPU/float64. The ROOT test was deselected because of the read lock already documented in the toy-generation review. Ruff passed on the sample module and the new regression file; `git diff --check` passed.
