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

`notebooks/12_cp_coefficients_closure.ipynb` demonstrates a complete generation-and-fit closure test with known non-zero `dx` and `dy`, independent positive/negative toys, cached likelihoods, one randomized start, and a simultaneous Minuit fit.
