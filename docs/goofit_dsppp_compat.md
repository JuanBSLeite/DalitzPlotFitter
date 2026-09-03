# DsPPP legacy GooFit compatibility

This branch contains opt-in compatibility conventions for reproducing the amplitude model used in `JuanBSLeite/GooFit` branch `DsPPP_ANA_Branch`, while keeping the standard DalitzPlotFitter conventions unchanged by default.

The compatibility layer addresses the main convention differences found in the code audit:

- `QMI(..., interpolation="linear-cartesian")` converts each magnitude/phase knot to real and imaginary parts and then interpolates those Cartesian components linearly in `s=m^2`. It also returns zero outside the knot range, matching the legacy `SplinePolar` behavior.
- `GooFitLegacyAngular()` implements the legacy spin factors used in the DsPPP analysis for `L=0,1,2`.
- When `GooFitLegacyAngular()` is selected, the parent Blatt-Weisskopf factor is evaluated with the bachelor momentum in the parent rest frame, as in the old GooFit implementation. The regular `CovariantAngular()` path retains the existing DalitzPlotFitter parent-barrier convention.
- `GooFitLegacyGounarisSakurai()` reproduces the literal derivative convention used in the old branch. `GounarisSakurai()` remains the default/current implementation.

A DsPPP-style model can therefore use:

```python
from dalitzplotfitter import (
    GooFitLegacyAngular,
    GooFitLegacyGounarisSakurai,
    QMI,
    Resonance,
)

legacy_angular = GooFitLegacyAngular()

s_wave = QMI(
    knots,
    magnitudes,
    phases,
    interpolation="linear-cartesian",
)

components = [
    Resonance(
        "pipi_S_qmi",
        (0, 1),
        s_coefficient,
        mass=1.0,
        width=0.0,
        spin=0,
        lineshape=s_wave,
        angular=legacy_angular,
        resonance_radius=1.5,
        parent_radius=5.0,
    ),
    Resonance(
        "rho770",
        (0, 1),
        rho_coefficient,
        mass=0.77526,
        width=0.1491,
        spin=1,
        lineshape=GooFitLegacyGounarisSakurai(),
        angular=legacy_angular,
        resonance_radius=1.5,
        parent_radius=5.0,
    ),
]
```

Use the same `legacy_angular` object for the RBW spin-1 and spin-2 resonances in the legacy reproduction model. The identical-pion symmetrization remains automatic.

## What this does not reproduce yet

This compatibility layer does not implement the narrow-omega detector-resolution treatment from the published Ds analysis, nor does it recreate the historical efficiency/background smoothing algorithm. For the latter, use the final smoothed ROOT maps from the analysis when available. These limitations should be kept in mind when comparing absolute likelihood values or fitted parameters with the publication.

The compatibility options are deliberately opt-in. Existing models that use `CovariantAngular()`, `GounarisSakurai()`, and ordinary QMI interpolation keep their previous behavior.
