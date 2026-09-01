# User-friendly analysis API

DalitzPlotFitter keeps the low-level classes available for validation and custom analyses, but common workflows can now use `FitSession`.

## Minimal signal fit

Low-level construction requires creating a `SignalPDF`, an `UnbinnedNLL` and a `Minimizer` explicitly. The high-level equivalent is:

```python
from dalitzplotfitter import FitSession

session = FitSession(model, data)
result = session.fit(simplex=True)
session.print_result(result)
session.print_fit_fractions(result)
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

A callable histogram/function can be supplied through `BackgroundSpec`:

```python
from dalitzplotfitter import BackgroundSpec, FitSession, Parameter

f_sig = Parameter("signal_fraction", 0.8, bounds=(0.0, 1.0))

session = FitSession(
    model,
    data,
    signal_fraction=f_sig,
    backgrounds=(
        BackgroundSpec("combinatorial", background_shape),
    ),
)
```

The session evaluates `background_shape` on both data and the model normalization grid and computes

```text
integral B(Phi) dPhi
```

automatically. If a veto is attached to the session, the same veto is applied to the background by default.

For multiple non-extended backgrounds:

```python
session = FitSession(
    model,
    data,
    signal_fraction=f_sig,
    backgrounds=(
        BackgroundSpec("comb", comb_shape, fraction=f_comb),
        BackgroundSpec("misid", misid_shape),  # remainder
    ),
)
```

Extended fits use `signal_yield=` and per-background `yield_=` instead.

## Constraints

```python
session = session.with_constraint(
    GaussianConstraint(f_sig, mean=0.82, sigma=0.03)
)
```

or pass a tuple of constraints directly when constructing the session.

## Plot helpers

```python
from dalitzplotfitter import plot_dalitz, plot_square_dalitz

plot_dalitz(data, x="s13", y="s23", title="Selected data")
plot_square_dalitz(
    data,
    mother_mass=model.channel.parent_mass,
    masses=model.channel.daughter_masses,
    pair=(0, 2),
)
```

These helpers remove the repeated `hist2d`, labels, ranges and Square-Dalitz conversion code from analysis notebooks.

## Design principle

`FitSession` is intentionally a composition layer. It does not replace `SignalPDF`, `PreparedAmplitudeCache`, `MultiBackgroundNLL`, `CPJointNLL` or `Minimizer`. Those remain the reference low-level API for advanced workflows and detailed validation.

## Good next ergonomic improvements

The next useful convenience layer would be:

1. declarative model construction from dictionaries/YAML/JSON;
2. automatic projection plots with signal/background component overlays;
3. a `CPFitSession` wrapping the two charge samples and `CPJointNLL`;
4. direct `model.fit(data, ...)` shorthand backed by `FitSession`;
5. a standard fit-report object/table including parameters, correlations, fit fractions and diagnostics;
6. automatic ROOT histogram discovery/inspection helpers;
7. optional pandas-compatible fit-result export.

These are ergonomics features and should be built on top of the validated numerical core rather than duplicating it.
