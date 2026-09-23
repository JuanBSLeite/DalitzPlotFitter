# Time-dependent neutral-meson Dalitz fits

`NeutralMesonMixing` and `TimeDependentDalitzNLL` implement an unbinned,
signal-only likelihood for `(Dalitz coordinates, decay time)`, conditional on
initial-flavour tag and, when supplied, per-event time uncertainty. Numerical
operations and derivatives are JAX; use the existing `Minimizer` directly.
This is a separate API from the charge-joint `CPJointNLL` for charged mesons.

## Conventions and reference

The numerical reference is Belle, **Phys. Rev. D 89, 091103 (2014)**,
[arXiv:1404.2412](https://arxiv.org/abs/1404.2412), specifically the **printed
rates in Eqs. (1–2)**. Let `u=t/tau`, `r=q/p`, `C=r*conj(A)*Abar`. The D0 rate is

```
exp(-u)/2 * [
    (|A|²+|r*Abar|²)*cosh(y*u)
  + (|A|²-|r*Abar|²)*cos(x*u)
  + 2*Re(C)*sinh(y*u) - 2*Im(C)*sin(x*u)
]
```

For D0bar, swap `A` and `Abar` and replace `r` with `1/r`. These interference
signs are part of the API; do not substitute a different eigenstate/g-minus
convention without transforming the parameters consistently.

`x`, `y` are fractions (Belle CP-conserving central values: `0.0056`, `0.0030`),
`q_over_p` is the positive magnitude, `phi` is its phase in radians, and `tau`
and all times share units (the example uses ps). Require `tau>0`, `|y|<1`,
`q_over_p>0`. Fields accept floats or `Parameter`s. Put physical bounds on
floating parameters. Invalid evaluations return infinite NLL.

## Preparing A and Abar

Use one `PreparedAmplitudeCache` containing all A components followed by all
Abar components. Both groups must refer to **the same final-state coordinates**
and the same integration measure; give every component a unique name. With
ordering `(KS, pi+, pi-)`, `s12=m²(KS pi+)` and `s13=m²(KS pi-)`. In the
no-direct-CPV convention, `Abar(s12,s13)=A(s13,s12)`. There is no identical-pion
symmetrization or folding: pi+ and pi- are distinct particles.

For a reflected callable, pass only the permuted raw invariants to the original
function, not its precomputed kinematic arrays. Retaining prepared arrays from
the unreflected point would silently evaluate the wrong amplitude. The runnable
example below implements this explicitly. Independently supplied A/Abar models
also support direct CP violation; reference phases and scales must be fixed to
remove unidentifiable directions, especially when fitting `q/p` simultaneously.

```python
from dalitzplotfitter import (
    Parameter, PreparedAmplitudeCache, NeutralMesonMixing,
    TimeDependentDalitzNLL, Minimizer,
)

# components_A / components_Abar: low-level AmplitudeComponent sequences.
# Both groups evaluated at the observed final-state point, not separate datasets.
x = Parameter("mix.x", 0.0056, bounds=(-0.1, 0.1), step=0.001)
y = Parameter("mix.y", 0.0030, bounds=(-0.1, 0.1), step=0.001)
parameters = (*amplitude_parameters, x, y)
cache = PreparedAmplitudeCache.prepare(
    (*components_A, *components_Abar),
    data=data.as_dict(),
    normalization_data=integration_sample.as_dict(),
    normalization_weights=integration_sample.weights,
    efficiency_normalization=efficiency_on_integration_sample,
    parameters=parameters,
    normalize_components=False,
)
cache.check_parameters(parameters)
nll = TimeDependentDalitzNLL(
    cache, len(components_A), times, tags,
    NeutralMesonMixing(x=x, y=y, tau=0.4103),
    efficiency=efficiency_on_data,
    time_range=(0.0, 4.0),
)
result = Minimizer(nll, parameters).fit()
```

`tags=+1` means initially D0; `-1` means initially D0bar. Conditional PDFs are
normalized separately for the two tags. The likelihood does not fit initial
production fractions, tag counts, or an extended yield. `wrong_tag` is the
posterior wrong-flavour probability **within the selected sample** (scalar,
event array, or scalar `Parameter`); it mixes already-normalized PDFs. A raw
preselection mistag rate with production/selection asymmetries requires its own
conversion to this conditional probability.

The new `cache.coherent_groups(values, groups)` method evaluates both amplitudes
and their 2x2 Hermitian overlap in one pass. For the two groups it retains
`integral(conj(A)*Abar)` as well as both diagonal integrals. Every Dalitz integral
uses `mean(sample.weights*f)`; temporal integrals have units of time. Fixed
lineshapes and overlap matrices stay cached when only mixing or coefficients
float; floating dynamics update the affected blocks through existing cache paths.

## Time acceptance and Gaussian resolution

The default uses exact finite-window or `[0,infinity)` time integrals, unit
acceptance, and perfect resolution. For factorized true-time acceptance and/or
a Gaussian resolution, provide quadrature explicitly:

```python
import numpy as np
import jax.numpy as jnp

nodes, weights = np.polynomial.legendre.leggauss(300)
true_time_max = 8.0  # ps; verify convergence for the lifetime/acceptance in use
nodes = true_time_max*(nodes+1)/2
weights = true_time_max*weights/2
nll = TimeDependentDalitzNLL(
    cache, len(components_A), measured_times, tags,
    NeutralMesonMixing(x=x, y=y, tau=0.4103),
    efficiency=efficiency_on_data,
    time_range=(-0.5, 4.0),
    sigma_t=event_time_errors,  # positive, fixed; scalar also accepted
    time_nodes=nodes, time_weights=weights,
    time_acceptance=lambda t, values: 1-jnp.exp(-t/0.15),
)
```

Temporal quadrature follows **sum(weights*f)**, not the Dalitz sample's mean
convention. With resolution, true time is nonnegative while measured time may
be negative. Gaussian CDF differences account for migration into/out of the
observed time window in the per-event normalization. Accumulation scans time
nodes to avoid materializing an events-by-nodes response matrix.

Acceptance acts on true time and must factorize from the Dalitz efficiency.
The PDF is conditional on sigma_t; it assumes the supplied conditional response
and acceptance describe that category. Correlated time/Dalitz acceptance,
non-Gaussian resolution mixtures, floating resolution calibration, backgrounds,
and automatic model serialization are not provided by this first API. Repeat
the quadrature with increased order and true-time endpoint, particularly for
narrow sigma_t or long lifetime tails. A quadrature endpoint truncates the
convolution; it is not an exact infinite-time Gaussian convolution.

## High-level session: TimeDependentFitSession

`TimeDependentFitSession` (`docs/user_friendly_api.md` "Design principle": a
composition layer, not a replacement) removes the A/Abar cache-building
boilerplate above. It composes one `DecayModel` (the D0/A amplitude), a
`PhaseSpaceSample` of observed Dalitz coordinates, parallel `times`/`tags`
arrays, and a `NeutralMesonMixing`:

```python
from dalitzplotfitter import NeutralMesonMixing, TimeDependentFitSession

session = TimeDependentFitSession(
    model, data, times, tags,
    NeutralMesonMixing(x=x, y=y, tau=0.4103),
    efficiency=efficiency_on_data,
    time_range=(0.0, 4.0),
)
result = session.fit()
session.report(result)
```

By default (no `abar_model`) it derives D0bar by reflection,
`Abar(s12,s13) = A(s13,s12)`, the same no-direct-CPV convention as the manual
example above -- it never re-derives or overrides the physics in
`TimeDependentDalitzNLL`/`NeutralMesonMixing`. Pass an independently built
`abar_model` (a second `DecayModel` sharing `model`'s daughter masses) for
direct CP violation; only its dynamics/coefficients are used, since the cache
is built once on `model`'s `data`/`normalization_sample` for both flavours.
`session.parameters` collects and deduplicates `Parameter`s from `model` (and
`abar_model`, if given) plus `mixing`. `fit(update_model=True)` also returns
the model(s) with fitted values baked in, via `model_with_fitted_values`.
Scope matches `TimeDependentDalitzNLL` exactly: signal-only, non-extended, no
backgrounds.

`session.plot_time_projection(result)` overlays each observed tag's decay-time
histogram against the model's exact Dalitz-integrated curve:

```python
session.plot_time_projection(result, bins=60, time_unit="ps")
```

The curve comes from `TimeDependentDalitzNLL.dalitz_integrated_time_pdf`, the
Dalitz marginal `p(t | tag) = integral_DP rate(t, s12, s13, tag) dPhi / norm`,
built from the same `cache.coherent_groups` overlap and `mixing.basis`/
`integrals` the fit itself normalizes against -- not a rendering MC sample, so
it shares that method's scope: unit temporal acceptance and perfect
resolution only (raises if `time_nodes`/`sigma_t`/`time_acceptance` are set),
and a scalar `wrong_tag` (a single curve needs one representative mistag
probability, not per-event values). A tag with zero observed events is
skipped.

`session.plot_projection(result, variable)` is the Dalitz-variable analogue,
one subplot per tag (D0 | D0bar), mirroring `CPFitSession.plot_projection`'s
two-population layout:

```python
session.plot_projection(result, "s13", bins=60)
```

A weighted phase-space MC sample renders the histogram; the density comes
from `TimeDependentDalitzNLL.tag_marginal_density`, the *time*-integrated
(over `time_range`) counterpart of `dalitz_integrated_time_pdf` --
`p(s12,s13 | tag) = integral_t rate(t, s12, s13, tag) dt / norm`, built from
the same `mixing.integrals` the fit itself normalizes against. It shares
`dalitz_integrated_time_pdf`'s scalar-`wrong_tag` requirement but not its
unit-acceptance/perfect-resolution one, since time is integrated out
analytically via `mixing.integrals` regardless of any observed-time
quadrature. `show_pulls=True` adds a pull panel below each tag's histogram,
same convention as `FitSession`/`CPFitSession`.

`plot_contour(result, "mix.x", "mix.y")` (not session-specific -- see the
"Plotting" section of `docs/catalog.md`) draws the published-style (x, y)
confidence-region plot: the profile-likelihood 68%/95% contours from
`Minuit.mncontour` around the fitted point, exactly the kind of figure a
mixing-parameter paper publishes (e.g. Fig. 4 of the Belle reference above).
`result` is any fitted `Minuit` object, i.e. what `session.fit()` returns
with the default `update_model=False`.

## Reproducible validation and scope of the Belle reproduction

```
JAX_PLATFORMS=cpu python benchmarks/benchmark_time_dependent.py --resolution 20
JAX_PLATFORMS=cpu python benchmarks/benchmark_time_dependent.py --resolution 20 --toy
pytest tests/test_time_dependent.py
```

The script uses physical D0 -> KS pi+ pi- kinematics and a **reduced** K*(892)-,
K*(892)+, rho model. The Asimov weights and optional sampled toy come from an
independent NumPy implementation of the printed Belle rates. It fits x,y with
JAX/Minuit, reports timing, and checks Asimov recovery at Belle's central mixing
values. The optional toy samples the finite Dalitz/time quadrature distribution;
it is a discrete approximation, not a new continuous public toy generator.
Neither this three-resonance model nor its fit fractions is Belle's full model.

Unit tests check both tags, overlap orientation, finite/infinite time integrals,
zero mixing, wrong tags, accepted and smeared normalization, independent SciPy
convolution, fixed-cache reuse, floating dynamics (chunked and unchunked), and
JAX gradients including x,y,tau,|q/p|,phase.

The paper provides a much fuller amplitude table, including K-matrix pi-pi and
K-pi S-waves, and describes detector resolution, efficiency and backgrounds.
Reproducing its full model requires a convention audit of those amplitudes and
external inputs cited by the paper. Reproducing the experimental best fit and
errors additionally requires event data and detector/background parameters not
supplied as numerical inputs here. This implementation validates the time
formalism and a controlled example; it does not certify reproduction of Belle's
experimental result.

The [full-component Belle notebook](../notebooks/benchmark/belle_2014_d0_kspipi_time_dependent.ipynb)
transcribes all 15 components and runs a fit-fraction audit and Asimov fit.
Its sizeable remaining FF discrepancies are recorded in the
[validation report](reviews/20260923_time_dependent_belle.md); do not treat it
as an already closed reproduction of Belle's amplitude conventions.
