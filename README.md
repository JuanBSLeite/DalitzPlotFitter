# DalitzPlotFitter

DalitzPlotFitter is a Python package under development for unbinned amplitude fits of three-body decays. The physics implementation follows **Laura++ conventions**, **JAX** evaluates amplitudes, likelihoods and gradients, **iminuit** performs minimization, and **phasespace** supplies weighted phase-space Monte Carlo.

## Architecture

There is one supported physics path:

```text
phasespace weighted MC
        -> (E, px, py, pz) four-momenta
        -> covariant kinematics
        -> Laura++ line shape
        -> Blatt-Weisskopf factors
        -> Laura++ Covariant angular factor
        -> Bose symmetrization when identical particles are present
        -> RealImag coefficient c = x + i y
        -> coherent amplitude A = sum_i c_i F_i
        -> weighted MC normalization
        -> JAX NLL + iminuit
```

AmpForm, QRules, TensorWaves, the former JAX phase-space generator, and the accept-reject `ToyGenerator` are not part of the code base.

## Complex coefficients

The supported coefficient parameterization is

```text
c_i = x_i + i y_i
```

through `RealImag`. `x` and `y` may be constants or fit `Parameter` objects. One amplitude coefficient is fixed to `1 + 0 i` to remove the arbitrary global scale and phase.

## Laura++ resonance dynamics

The first complete resonance component is `LauraCovariantRBW`:

```text
F = R(m) X_L(p* r_parent) X_L(q r_res) T_L^Covariant
```

with

```text
R(m) = 1 / (m0^2 - m^2 - i m0 Gamma(m))
Gamma(m) = Gamma0 (q/q0)^(2L+1) (m0/m) X_L(q r_res)^2.
```

The angular dependence uses the **Laura++ Covariant formalism** for `L=0..4`. Event-wise quantities are calculated directly from four-momenta:

```text
p*          bachelor momentum in the parent rest frame
p           bachelor momentum in the resonance rest frame
q           selected resonance-daughter momentum in the resonance rest frame
cos(theta)  angle between that daughter and the bachelor in the resonance rest frame
```

The selected daughter fixes the sign convention for odd-spin amplitudes. Tests verify this behavior explicitly.

See `docs/lineshapes.md` for the formulas.

## Weighted phase-space Monte Carlo

`PhasespaceMC` wraps the `phasespace` package:

```python
from dalitzplotfitter import PhasespaceMC

mc = PhasespaceMC(
    mother_mass=1.86966,
    masses=(0.13957, 0.13957, 0.13957),
)
sample = mc.generate(1_000_000, seed=2027)
```

`phasespace` four-vectors are converted once from `(px, py, pz, E)` to the internal `(E, px, py, pz)` convention. DalitzPlotFitter requests raw phase-space weights with `normalize_weights=False`; these weights are used directly in Monte Carlo integration.

For example,

```text
N(theta) proportional to sum_k w_PS(k) |A(x_k; theta)|^2.
```

TensorFlow is therefore confined to event generation inside `phasespace`; no TensorFlow object enters the JAX likelihood.

## Pseudo-data generation

Unweighted pseudo-data are produced from a larger weighted candidate pool. For generated parameters `theta_gen`, define

```text
w_target(k) = w_PS(k) |A(x_k; theta_gen)|^2.
```

`weighted_resample()` samples candidate indices according to these weights and returns events with unit weights. The unbinned fit therefore receives ordinary unweighted pseudo-data rather than treating weighted MC candidates as independent observations.

## D+ -> pi- pi+ pi+ reference ordering

The reference ordering is

```text
p1 = pi-
p2 = pi+_1
p3 = pi+_2
```

so

```text
s12 = m2(pi- pi+_1)
s13 = m2(pi- pi+_2)
s23 = m2(pi+_1 pi+_2).
```

Identical-pion symmetrization coherently adds the `(p1,p2)` and `(p1,p3)` resonance pairings with their corresponding bachelor assignments. `BoseSymmetrizedAmplitude` provides this operation before multiplication by the external complex coefficient.

The target reference model is

```text
A = c_rho F_rho + c_f0 F_f0 + c_NR
```

with Gounaris-Sakurai planned for `rho(770)`, Flatte planned for `f0(980)`, and the Covariant angular formalism for nonzero spin.

## E791 Fit 2 generation example

`notebooks/01_e791_dplus_fit2_generation.ipynb` implements a generation example based on Fit 2 of

```text
E. M. Aitala et al. (E791), arXiv:hep-ex/0007028v2.
```

The notebook uses the published Fit 2 central magnitudes and phases for

```text
sigma(500) pi+
rho0(770) pi+
non-resonant
f0(980) pi+
f2(1270) pi+
f0(1370) pi+
rho0(1450) pi+
```

with explicit Bose symmetrization, the Laura++ Covariant angular convention, a 1,000,000-event weighted `phasespace` candidate pool, and resampling to 100,000 unweighted pseudo-data events. It includes plots of raw phase space, the weighted model Dalitz density, the resampled Dalitz plot, the symmetrized `m2(pi+pi-)` projection, individual component intensities and the net interference.

For `f0(980)` the example uses the single-channel RBW alternative quoted in the E791 companion study (`m0 = 0.975 GeV`, `Gamma0 = 0.044 GeV`), which that study reports as giving essentially indistinguishable fractions and phases from its coupled-channel parametrization. This keeps the example inside the currently implemented Laura++-style RBW path until the project-owned Flatte component is added.

## Normalization and caching

For coefficient-only fits, component values are cached on the data and normalization samples. The normalization matrix is

```text
M_ij = (1/N_MC) sum_k w_PS(k) F_i*(x_k) F_j(x_k),
N(c) = c^dagger M c.
```

The overall phase-space normalization constant is independent of fit parameters and therefore irrelevant to the NLL minimum.

The reference validation scale is:

```text
fit pseudo-data:      100,000 events
normalization MC:   1,000,000 weighted events
```

For each floating `RealImag` coordinate, closure is defined by

```text
abs((value_gen - value_fit) / sigma_fit) < 1
```

using HESSE uncertainties with the NLL convention `errordef=0.5`.

## Current dynamics roadmap

1. Laura++ relativistic Breit-Wigner — implemented;
2. Laura++ Covariant angular formalism — implemented for `L=0..4`;
3. weighted `phasespace` MC — implemented;
4. weighted pseudo-data resampling — implemented;
5. complete covariant RBW component — implemented;
6. explicit identical-particle Bose symmetrization — implemented;
7. E791 Fit 2 generation notebook — implemented;
8. new end-to-end closure on the Laura++/phasespace path — next;
9. Gounaris-Sakurai for `rho(770)`;
10. Flatte for `f0(980)`;
11. LASS and K-matrix after the simpler models pass closure/reference tests.

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

Double precision is recommended for amplitude fits:

```python
from dalitzplotfitter import enable_x64
enable_x64()
```

## Reference

The resonance, Blatt-Weisskopf and Covariant angular conventions follow:

J. Back et al., **Laura++: a Dalitz plot fitter**, Computer Physics Communications 231 (2018) 198-242, arXiv:1711.09854.
