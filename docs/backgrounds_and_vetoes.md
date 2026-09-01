# Multiple background categories and veto maps

DalitzPlotFitter supports arbitrary named background categories and Laura++-style Dalitz vetoes.

## Multiple background categories

For a non-extended fit, the total signal fraction remains the primary mixture parameter:

```text
p(x) = f_sig S(x) + (1 - f_sig) sum_k w_k B_k(x)
```

where every `B_k` is independently normalized and the relative background weights satisfy

```text
sum_k w_k = 1.
```

For `N` background categories, the first `N-1` categories may carry relative `fraction` parameters and the final category is the remainder. This avoids introducing a redundant normalization parameter.

Example:

```python
comb = BackgroundCategory(
    "combinatorial",
    values=comb_data,
    normalization=comb_norm,
    fraction=Parameter("f_comb", 0.6, bounds=(0.0, 1.0)),
)
partial = BackgroundCategory(
    "partially_reconstructed",
    values=partial_data,
    normalization=partial_norm,
)

nll = MultiBackgroundNLL(
    signal_density=lambda values: signal_pdf(values),
    backgrounds=(comb, partial),
    signal_fraction=Parameter("f_sig", 0.8, bounds=(0.0, 1.0)),
)
```

The mixture is then

```text
p = f_sig S + (1-f_sig) [f_comb B_comb + (1-f_comb) B_partial].
```

### Extended mode

In an extended fit every component has its own expected yield:

```text
lambda(x) = N_sig S(x) + sum_k N_k B_k(x)
```

and

```text
NLL = N_sig + sum_k N_k - sum_events log(lambda(x)).
```

Each `BackgroundCategory` then uses `yield_=` rather than `fraction=`.

## Multiple backgrounds in CP fits

`CPJointNLL` accepts `CPBackgroundCategory` objects. Each category has separate positive- and negative-charge shapes and normalization integrals but is normalized in the joint `(Dalitz, charge)` sample space:

```text
B_k(phi,+) = B^raw_{k,+}(phi) / (J_{k,+}+J_{k,-})
B_k(phi,-) = B^raw_{k,-}(phi) / (J_{k,+}+J_{k,-}).
```

The non-extended convention remains

```text
p_q = f_sig S_q + (1-f_sig) sum_k w_k B_{k,q}.
```

In extended mode each CP background category has an independent global yield.

The older single-background arguments of `CPJointNLL` are retained for compatibility with existing notebooks.

## Laura++-style veto maps

A veto is represented by a binary acceptance function

```text
V(phi) = 1  accepted
V(phi) = 0  vetoed.
```

For the signal PDF,

```text
P_sig(phi) = V(phi) epsilon(phi) |A(phi)|^2
             / integral V epsilon |A|^2 dPhi.
```

Thus the same veto must be applied to the event sample and to all normalization integrals.

### Mass-window vetoes

`MassWindowVeto` follows the Laura++ `addMassVeto` convention: bounds are specified in invariant mass in GeV, not mass squared.

```python
charm_veto = MassWindowVeto((0, 2), 1.84, 1.89)
```

This rejects points with

```text
1.84 <= m13 <= 1.89 GeV.
```

Several vetoes can be combined:

```python
veto = CompositeVeto(
    MassWindowVeto((0, 2), 1.84, 1.89),
    MassWindowVeto((1, 2), 3.00, 3.20),
)
```

Arbitrary accepted regions can be represented by `FunctionalVeto`.

### Applying vetoes consistently

For data or generated phase-space samples:

```python
accepted_sample = veto.apply(sample)
```

For a signal PDF:

```python
pdf = SignalPDF(
    intensity=intensity,
    integrator=integrator,
    efficiency=efficiency,
    veto=veto,
)
```

For a background shape, wrap it with the same veto:

```python
vetoed_background = VetoedDensity(background, veto)
```

The background normalization must then be computed from `vetoed_background` on the normalization sample. This ensures signal, every background category, generated toys and fitted data all use exactly the same accepted Dalitz region.

## Relation to SCF

Vetoes act on the accepted reconstructed phase space. When SCF is enabled, the SCF migration map should be constructed for the same accepted region, or vetoed reconstructed bins should carry zero accepted probability. The SCF machinery and veto maps are intentionally kept as separate objects so detector migration and analysis selection remain independently testable.
