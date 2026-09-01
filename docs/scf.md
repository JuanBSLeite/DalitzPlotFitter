# Self-cross-feed (SCF) migration

DalitzPlotFitter supports Laura++-style self-cross-feed (SCF), also called
misreconstructed-signal migration.  SCF is treated as part of the signal model,
not as an incoherent physics background.

## True and reconstructed Dalitz coordinates

For a true Dalitz point `t` and reconstructed point `r`, define the true
selected-signal density

```text
rho(t) = epsilon(t) |A(t)|^2.
```

The SCF fraction `f_SCF(t)` is defined in the **true** Dalitz plane.  The signal
is split into

```text
correctly reconstructed: (1 - f_SCF(t)) rho(t)
SCF:                     f_SCF(t) rho(t).
```

The SCF part is redistributed by a response matrix

```text
M[t, r] = P(reconstructed bin r | true bin t).
```

Every populated true-bin row must therefore satisfy

```text
sum_r M[t, r] = 1.
```

This row-normalized convention is the one used by `SquareDalitzSCFMap`.

## Discrete Square-Dalitz formula

Laura++ implements the resolution model in Square-Dalitz bins.  For true bin
`t` and reconstructed bin `r`, let

```text
DeltaOmega_k = |J_k| Delta m' Delta theta'
```

be the ordinary Dalitz phase-space area represented by that uniform
Square-Dalitz bin.  The migrated SCF density in reconstructed bin `r` is

```text
rho_SCF(r) = 1/DeltaOmega_r
             sum_t M[t,r] f_SCF(t) rho(t) DeltaOmega_t.
```

The ratio `DeltaOmega_t / DeltaOmega_r` is the discrete Jacobian ratio that is
required when a migration histogram is defined in uniform Square-Dalitz
coordinates but the PDF is a density with respect to ordinary Dalitz phase
space.

`SquareDalitzSCFMap.smeared_bin_density` implements this expression directly.

## Full reconstructed signal PDF

At a reconstructed Dalitz point `r`, the model is

```text
rho_reco(r) = (1 - f_SCF(r)) epsilon(r) |A(r)|^2
              + rho_SCF(r).
```

For the correctly reconstructed term the true and reconstructed coordinates
coincide, hence evaluating `f_SCF` at `r` is equivalent to evaluating it at the
true position.

Since the migration distribution of every true bin is normalized, migration
conserves signal probability:

```text
integral rho_reco(r) dr = integral epsilon(t) |A(t)|^2 dt.
```

Therefore the normalized signal PDF remains

```text
P_sig(r) = rho_reco(r) / integral epsilon(t) |A(t)|^2 dt.
```

This is implemented by `SCFSignalPDF`.

## API

```python
import jax.numpy as jnp
from dalitzplotfitter import SCFSignalPDF, SquareDalitzSCFMap

scf_map = SquareDalitzSCFMap(
    migration=migration_matrix,       # shape (Ntrue, Nreco)
    scf_fraction=scf_fraction,        # shape (Ntrue,)
    mother_mass=channel.parent_mass,
    masses=channel.daughter_masses,
    bins_mprime=40,
    bins_thetaprime=40,
    pair=(0, 2),
)

pdf = SCFSignalPDF(
    intensity=intensity,
    integrator=integrator,
    efficiency=efficiency,
    scf_map=scf_map,
)
```

The matrix and `f_SCF` map must use the same Square-Dalitz binning.  This is
checked at construction time.

## Important convention when importing Laura++ histograms

DalitzPlotFitter stores a two-dimensional matrix with the explicit orientation

```text
axis 0 = true bin
axis 1 = reconstructed bin.
```

If an external histogram uses the opposite orientation it must be transposed
before constructing `SquareDalitzSCFMap`.  A useful validation is that every
true-bin row with nonzero SCF fraction sums to one.

## Validation tests

The test suite verifies that:

1. every migration row is normalized;
2. SCF probability mass is conserved under arbitrary migrations;
3. an identity migration reproduces the original signal density;
4. a true bin with nonzero SCF fraction cannot have an empty migration row.
