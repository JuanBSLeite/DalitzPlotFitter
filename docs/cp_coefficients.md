# Direct-CP coefficient parameterization

DalitzPlotFitter provides `CPRealImag` for simultaneous fits of charge-conjugate Dalitz samples. It is a Cartesian extension of `RealImag` and is designed to keep the CP-averaged amplitude and the direct-CP difference in one shared parameter set.

For charge label `q = +1` or `q = -1`,

```text
c_q = (x + q dx) + i (y + q dy).
```

Therefore

```text
c_plus  = (x + dx) + i (y + dy)
c_minus = (x - dx) + i (y - dy).
```

`x` and `y` are the CP-averaged Cartesian coefficient components. `dx` and `dy` are CP-odd differences. Setting

```text
dx = 0
dy = 0
```

recovers the CP-conserving case `c_plus = c_minus`.

All four entries may be ordinary numbers or `Parameter.coefficient(...)` objects. The same `Parameter` instances should be shared between the two charge models:

```python
x = Parameter.coefficient("rho.x", 0.8, owner="rho")
y = Parameter.coefficient("rho.y", -0.2, owner="rho")
dx = Parameter.coefficient("rho.dx", 0.1, owner="rho")
dy = Parameter.coefficient("rho.dy", 0.05, owner="rho")

cp = CPRealImag(x, y, dx, dy)
coefficient_plus = cp.for_charge(+1)
coefficient_minus = cp.for_charge(-1)
```

A simultaneous fit is formed by preparing one likelihood/cache for each charge and combining them with `SimultaneousNLL`. The likelihood terms receive the same flat parameter mapping, so CP-even and CP-odd parameters are fitted together.

When only coefficient parameters float, the existing amplitude cache is fully reused: the dynamical basis and normalization matrix are not recalculated during minimization. Only the charge-dependent coefficient vectors and the quadratic normalization `c^dagger M c` change.

At least one complex amplitude convention must still be fixed, exactly as in an ordinary amplitude fit. For example, a reference component can be fixed to `1 + 0i`; otherwise the overall complex scale/phase produces degeneracies.

## LHCb-inspired B+ -> K+ pi+ pi- benchmark

`notebooks/12_cp_coefficients_closure.ipynb` uses the same Cartesian CP convention as the 2026 LHCb amplitude analysis of `B+ -> K+ pi+ pi-` (arXiv:2608.12612, 2608.12613 and 2608.12614).

The first benchmark is deliberately a truncated non-S-wave model containing:

- `K*(892)0`, fixed to `1 + 0i` as the common reference;
- `rho(770)0`;
- `K2*(1430)0`;
- `f2(1270)`;
- `rho3(1690)0`.

For every non-reference component, the notebook reconstructs charge-dependent complex coefficients from the published Isobar-model central values of fit fraction, quasi-two-body CP asymmetry, average strong phase, and `delta_minus - delta_plus`. The reconstructed `c_plus` and `c_minus` are then converted into `x`, `y`, `dx`, and `dy` and used as the truth parameters.

The notebook generates independent `B+` and `B-` pseudo-data samples, performs one simultaneous fit with one randomized start, and compares both the Cartesian parameters and the reconstructed physical observables

```text
A_CP = (|c_minus|^2 - |c_plus|^2) / (|c_minus|^2 + |c_plus|^2)
Delta phi = arg(c_minus / c_plus).
```

This benchmark is intended as a fitter closure test, not as an exact reproduction of the full LHCb experimental likelihood. The published model also contains several additional resonances and substantial `K pi` and `pi pi` S-wave contributions. Those will be added only after the non-S-wave CP closure is stable.
