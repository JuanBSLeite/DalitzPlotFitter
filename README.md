# DalitzPlotFitter

DalitzPlotFitter is a Python package under development for unbinned amplitude fits of three-body decays. The primary physics convention is being aligned directly with **Laura++**, while **JAX** is used for numerical amplitude evaluation, likelihoods and gradients and **iminuit** is used for minimization.

## Current physics strategy

The primary model path is now

```text
phasespace weighted MC
        -> four-momenta in the parent rest frame
        -> Dalitz invariants and Laura++ covariant kinematics
        -> Laura++ line shape + Blatt-Weisskopf + Covariant angular term
        -> complex RealImag coefficient
        -> coherent amplitude
        -> cached MC normalization
        -> JAX NLL + iminuit
```

QRules/AmpForm remain useful during the transition for topology validation and reference comparisons, but production dynamical PDFs should follow the Laura++ conventions implemented explicitly by DalitzPlotFitter.

## Coherent amplitude convention

The signal amplitude is

```text
A(x) = sum_i c_i F_i(x)
```

where the preferred CP-conserving coefficient convention is

```text
c_i = x_i + i y_i
```

through `RealImag`. The reference amplitude fixes one coefficient to `1 + 0 i` to remove the arbitrary global scale and phase.

## Laura++ resonance components

The first complete project-owned resonance component is `LauraCovariantRBW`. It evaluates

```text
F = R(m) X_L(p* r_parent) X_L(q r_res) T_L^Covariant
```

with the Laura++ relativistic Breit-Wigner

```text
R(m) = 1 / (m0^2 - m^2 - i m0 Gamma(m))
```

and

```text
Gamma(m) = Gamma0 (q/q0)^(2L+1) (m0/m) X_L(q r_res)^2.
```

The angular term uses the **Laura++ Covariant formalism**, not Zemach or the AmpForm helicity factor. `covariant_angular_factor()` and the numerical `covariant_spin_factor()` implement the published Laura++ expressions for `L=0..4`.

The event-wise inputs are calculated directly from four-vectors:

```text
p*        bachelor momentum in the parent rest frame
p         bachelor momentum in the resonance rest frame
q         chosen resonance-daughter momentum in the resonance rest frame
cos(theta) angle between the chosen daughter and bachelor in the resonance rest frame
```

The chosen daughter fixes the sign convention for odd-spin amplitudes. Tests verify that exchanging equal-mass resonance daughters flips `cos(theta)` and the complete `L=1` amplitude sign.

See `docs/lineshapes.md` for the detailed formulas and implementation plan.

## Weighted phase-space Monte Carlo

The primary MC generator is now the external `phasespace` package. `PhasespaceMC` wraps it and immediately converts its TensorFlow output to JAX arrays.

```python
from dalitzplotfitter import PhasespaceMC

mc = PhasespaceMC(
    mother_mass=1.86966,
    masses=(0.13957, 0.13957, 0.13957),
)
sample = mc.generate(1_000_000, seed=2027)
```

`phasespace` produces four-vectors in `(px, py, pz, E)` order. The wrapper converts them once to the DalitzPlotFitter convention `(E, px, py, pz)` and calculates `s12`, `s13` and `s23`.

For Monte Carlo integration DalitzPlotFitter requests

```text
normalize_weights=False
```

from `phasespace`. This is important: independently normalizing each generated batch to its own maximum would make weights from different batches incompatible. The raw phase-space weights are retained and used in normalization sums such as

```text
N(theta) proportional to sum_k w_PS(k) |A(x_k; theta)|^2.
```

TensorFlow is therefore confined to MC generation; no TensorFlow object enters the JAX likelihood or minimization.

## Covariant kinematics API

`covariant_kinematics(daughter, partner, bachelor)` returns

```text
resonance_mass
p_star
p
q
cos_theta
```

for arrays of four-vectors stored as `(E, px, py, pz)`. Lorentz boosts are performed numerically with JAX and are independent of the MC source, so the same code can be used for generated events and real data.

## D+ -> pi- pi+ pi+ reference ordering

The reference ordering remains

```text
1 = pi-
2 = pi+_1
3 = pi+_2
```

and

```text
s12 = m2(pi- pi+_1)
s13 = m2(pi- pi+_2)
s23 = m2(pi+_1 pi+_2).
```

For a `rho(770)0` in the `(pi-, pi+_1)` pair, the other `pi+_2` is the bachelor. The identical-pion contribution with `(pi-, pi+_2)` must be added coherently with the corresponding daughter/bachelor assignment.

The intended physics model is

```text
A = c_rho F_rho + c_f0 F_f0 + c_NR
```

with Gounaris-Sakurai planned for `rho(770)`, Flatte planned for `f0(980)`, and the Laura++ Covariant angular term used for nonzero spin.

## Fit parameters and caching

For coefficient-only fits the complete component amplitudes `F_i(x)` are evaluated once on the data and normalization samples. The normalization matrix

```text
M_ij = integral epsilon(x) F_i*(x) F_j(x) dPhi
```

is cached, so

```text
N(c) = c^dagger M c.
```

When a dynamical parameter is eventually floated, only its owning component and the affected normalization-matrix row/column should be invalidated.

## Numerical convention

The signal PDF is

```text
p_sig(x | theta) = epsilon(x) |A(x; theta)|^2
                   ------------------------------------
                   integral epsilon(x) |A(x; theta)|^2 dPhi
```

The preferred reference normalization sample contains **1,000,000 weighted phase-space events**. The reference fit/toy sample target remains **100,000 events**.

For closure of `RealImag` coefficients, generated and fitted parameters are considered compatible when each fitted coordinate satisfies

```text
abs((x_gen - x_fit) / sigma_x_fit) < 1
abs((y_gen - y_fit) / sigma_y_fit) < 1
```

using HESSE uncertainties with the NLL convention `errordef=0.5`.

## Transition status

`ThreeBodyPhaseSpace` and the deterministic-envelope `ToyGenerator` are retained temporarily because existing tests and examples still exercise them. They are no longer the intended primary MC path. The closure notebook and end-to-end closure test will be migrated to `PhasespaceMC` and weighted/resampled MC after the new Laura++ covariant component has passed its lower-level validation tests.

## Planned native dynamics

1. Laura++ relativistic Breit-Wigner — implemented;
2. Laura++ Covariant angular term — implemented for `L=0..4`;
3. weighted `phasespace` MC wrapper — implemented;
4. complete Laura++ covariant RBW component — implemented, validation in progress;
5. Gounaris-Sakurai — next for `rho(770)`;
6. Flatte — next for `f0(980)`;
7. LASS and K-matrix after the simpler models pass closure/reference tests.

## Installation

```bash
git clone https://github.com/JuanBSLeite/DalitzPlotFitter.git
cd DalitzPlotFitter
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

For amplitude fits, double precision is recommended:

```python
from dalitzplotfitter import enable_x64
enable_x64()
```

## Reference

The amplitude, Blatt-Weisskopf and Covariant angular conventions are based on:

J. Back et al., **Laura++: a Dalitz plot fitter**, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.
