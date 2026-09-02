# User-friendly analysis API

DalitzPlotFitter keeps the low-level classes available for validation and custom analyses, while common workflows can use `FitSession` and `CPFitSession`.

## Minimal signal fit

```python
from dalitzplotfitter import FitSession

session = FitSession(model, data)
result = session.fit(simplex=True)
session.report(result)
session.plot_projection(result, "s13")
```

The fit parameters are collected automatically from the amplitude model.

## ROOT file to fit

```python
session = FitSession.from_root(
    model,
    "data.root",
    "DecayTree",
    s12="S12",
    s13="S13",
    s23="S23",
)
result = session.fit()
```

`FitSession.from_root` forwards ROOT branch, cut and entry-range options to `read_phase_space_sample`.

## Efficiency and veto

```python
session = FitSession(
    model,
    data,
    efficiency=efficiency,
    veto=veto,
)
```

The same efficiency and veto are included in the signal numerator and deterministic normalization.

## Backgrounds without manual normalization

```python
from dalitzplotfitter import BackgroundSpec, FitSession, Parameter

f_sig = Parameter("signal_fraction", 0.8, bounds=(0.0, 1.0))

session = FitSession(
    model,
    data,
    signal_fraction=f_sig,
    backgrounds=(BackgroundSpec("combinatorial", background_shape),),
)
```

The session evaluates the shape on data and the model normalization grid and computes its normalization automatically. If a veto is attached to the session, it is also applied to the background by default.

For multiple non-extended backgrounds:

```python
session = FitSession(
    model,
    data,
    signal_fraction=f_sig,
    backgrounds=(
        BackgroundSpec("comb", comb_shape, fraction=f_comb),
        BackgroundSpec("misid", misid_shape),
    ),
)
```

The final background remains the remainder category. Extended fits use `signal_yield=` and per-background `yield_=` instead.

## Constraints

```python
session = session.with_constraint(
    GaussianConstraint(f_sig, mean=0.82, sigma=0.03)
)
```

## Automatic fit report

```python
report = session.report(result)
```

The returned dictionary contains fit validity, NLL, EDM, function-call count, parameter values/errors, optional correlation matrix and optional fit fractions. The same information is printed in a compact form.

## Automatic projections

```python
session.plot_projection(result, "s13")
session.plot_projection(result, "s23", show_components=True)
session.plot_projection(result, "s13", log_scale=True)
```

Data are shown as black circular points with statistical error bars, while the fitted signal/background components and total fit are drawn as lines. For `s12`, `s13`, and `s23`, the vertical-axis label is generated automatically from the actual uniform bin width, for example `Candidates / 0.25 GeV^2`. `log_scale=True` changes the projection y axis to logarithmic scale.

Fit and PDF normalization always remain deterministic and use the configured Gauss--Legendre or Square-Dalitz quadrature. The quadrature nodes are **not** histogrammed directly for display, because a deterministic integration grid can alias strongly when projected onto arbitrary one-dimensional histogram bins. Instead, `plot_projection()` generates a weighted phase-space Monte Carlo sample only for rendering the fitted curves. This does not modify the NLL, fitted parameters, normalization integrals, or fit fractions.

The default rendering sample contains 250000 phase-space points and is deterministic for a fixed seed. It can be adjusted if needed:

```python
session.plot_projection(
    result,
    "s13",
    projection_size=500_000,
    projection_seed=123,
)
```

## Simultaneous direct-CP fits

The manual `PreparedAmplitudeCache + CPJointNLL + Minimizer` assembly can be replaced by:

```python
from dalitzplotfitter import CPFitSession

session = CPFitSession(
    plus_model,
    minus_model,
    plus_data,
    minus_data,
)
result = session.fit()
session.report(result)
session.plot_projection(result, "s13")
```

Shared `Parameter` objects appearing in the B+ and B- models are collected only once.

### CP efficiency and veto

For a charge-symmetric acceptance:

```python
session = session.with_efficiency(efficiency)
session = session.with_veto(veto)
```

The same object is applied to both charges. Charge-specific objects can be supplied as the second argument.

The acceptance is folded into both data numerators and the two normalization caches.

### CP backgrounds

```python
from dalitzplotfitter import CPBackgroundSpec

session = CPFitSession(
    plus_model,
    minus_model,
    plus_data,
    minus_data,
    signal_fraction=f_sig,
    backgrounds=(
        CPBackgroundSpec("combinatorial", background_shape),
    ),
)
```

If only `plus_shape` is supplied, the same shape is used for B-. Supplying `minus_shape=` allows charge-dependent background shapes. Each category is normalized automatically in the joint charge-Dalitz space.

Multiple categories follow the same remainder convention as `CPJointNLL`. Extended fits use the existing signal/background yield convention; independent B+/B- production/detection yield nuisance parameters remain intentionally outside this convenience layer for now.

### CP reports and projections

```python
session.report(
    result,
    include_fit_fractions=True,
    acceptance_weighted_fractions=True,
)

session.plot_projection(result, "s13")
session.plot_projection(result, "s23", log_scale=True)
```

The CP projections use two weighted phase-space MC rendering samples, but their signal and background weights are normalized **jointly across B+ and B-**. The plots therefore retain the same common global normalization as `CPJointNLL`; B+ and B- are not normalized independently and an integrated charge asymmetry remains visible.

## Plot helpers

```python
from dalitzplotfitter import plot_dalitz, plot_square_dalitz

plot_dalitz(data, x="s13", y="s23", title="Selected data")
plot_dalitz(data, x="s13", y="s23", log_scale=True)

plot_square_dalitz(
    data,
    mother_mass=model.channel.parent_mass,
    masses=model.channel.daughter_masses,
    pair=(0, 2),
    log_scale=True,
)
```

For two-dimensional Dalitz and Square-Dalitz plots, `log_scale=True` applies logarithmic normalization to the color scale rather than changing either coordinate axis.

## ROOT CP input

```python
session = CPFitSession.from_root(
    plus_model,
    minus_model,
    "Bplus.root", "DecayTree",
    "Bminus.root", "DecayTree",
    plus_root_kwargs={"s12":"S12", "s13":"S13", "s23":"S23"},
    minus_root_kwargs={"s12":"S12", "s13":"S13", "s23":"S23"},
)
```

## Design principle

`FitSession` and `CPFitSession` are composition layers. They do not replace `SignalPDF`, `PreparedAmplitudeCache`, `MultiBackgroundNLL`, `CPJointNLL` or `Minimizer`. Those remain the low-level API for advanced workflows and detailed validation.

## Remaining ergonomic improvements

Useful future additions are:

1. declarative model construction from dictionaries/YAML/JSON;
2. direct `model.fit(data, ...)` shorthand backed by `FitSession`;
3. automatic ROOT file/tree/histogram inspection;
4. optional pandas-compatible result export;
5. component-level amplitude projection overlays and standardized pull/residual panels.

These should continue to sit on top of the validated numerical core rather than duplicate it.
