# Monte Carlo normalization

Pass a `PhaseSpaceSample` as `DecayModel(..., normalization_sample=sample)` to
select `toy-mc`. The model reuses that sample; it does not draw new integration
points during minimization. Every integral uses `mean(sample.weights * f)`.

For samples drawn from a known normalized proposal density `q` in the desired
measure, use:

```python
normalization = events.with_importance_weights(q(events.as_dict()))
model = DecayModel(channel, components, normalization_sample=normalization)
```

This replaces existing weights with `1/q`; it does not multiply them. The proposal
must cover every region where the integrand contributes. Neither coverage nor
normalization of q can be inferred from the sampled points. Relative q gives
only relative integrals. Unit weights are appropriate for uniform phase-space
sampling up to the volume convention, not for arbitrary signal toys. MC already
selected by detector efficiency also needs the appropriate proposal convention;
do not count efficiency twice.

`PhaseSpaceMC.generate()` already supplies weights for its proposal. Its measure
is `ds12 ds13 / (128*pi**3*M**2)`; deterministic grids use `ds12 ds13`.
Multiply MC weights by `128*pi**3*M**2` when absolute integrals in the grid measure
are required. Keep the same measure across signal, backgrounds and charges.

`generate_stratified_invariants()` and `generate_stratified_phase_space()` now
return integration weights by default, including `1/(n_cells*p_cell)`.
Cell probabilities are normalized internally and must be finite, non-negative,
and have positive sum. Integration requires strictly positive probability in
every cell. `integration_weights=False` returns the proposal weights used by the
accept-reject algorithm; those weights must not be integrated directly. The
internal toy generator explicitly selects this proposal mode.

## Applying selection

The simplest integration selection is to keep the full sample and multiply the
integrand by the veto. To physically remove events while preserving the integral:

```python
selected = sample.select_for_integration(mask)
# Equivalent with a veto map:
selected = veto.apply(sample, for_integration=True)
```

Retained weights are multiplied by `N_retained/N_original`. An empty or entirely
zero-weight result is rejected because it cannot normalize a PDF. Ordinary
`take()` and `veto.apply(sample)` remain data selections and preserve the original
weights; they must not be confused with integration selections.

## Validation and precision

External samples are validated at model construction and when explicitly passed
to `pdf` or `prepare_cache`. Invariants and weights must be finite vectors of the
same nonzero length. Weights must be non-negative with at least one positive
entry. Signed subtraction samples are not supported as normalization samples.
If four-momenta are supplied, all three finite `(N, 4)` arrays are required.
These checks do not certify the proposal or physical kinematic domain.

Validate convergence by increasing sample size and using independent integration
seeds, keeping each sample fixed during its fit. Compare with converged quadrature
where possible. Integration uncertainty is not automatically propagated into the
iminuit covariance.
