# Initial four-body amplitude analysis

The new API composes the existing JAX amplitude cache, Hermitian normalization
matrix, signal PDF, likelihood and minimizers. It supports a **spin-zero parent
and four spin-zero daughters**, with intermediate integer spins and orbital
angular momenta 0 through 4. The three-body API and its defaults are unchanged.

## Architecture audit and boundaries

The implementation was designed after inspecting the following assumptions:

| Layer | Existing three-body assumption | Four-body extension |
|---|---|---|
| Channel (`decay.py`) | Exactly three daughters; resonance pair plus one bachelor | `NBodyDecayChannel` declares masses and optional identity labels; `FourBodyDecayModel` requires four daughters |
| Events (`kinematics/sample.py`) | `s12,s13,s23,p1,p2,p3`; third invariant derived from two | `EventSample` protocol; `NBodySample` stores `(events,N,4)` momenta, weights, subset invariants and orientation |
| Kinematics | Dalitz boundary and Square-Dalitz maps assume dimension two | Five pair-chain or cascade coordinates, with signed azimuth and a canonical inverse map |
| Amplitudes (`decay.py`, `dynamics/`) | One isobar paired with a bachelor, three-body angular functions | `Isobar`, `PairChain`, `CascadeChain`, using the existing `lineshape(mass, context)` contract |
| Normalization (`integration/`, `amplitude/cache.py`) | Built-in quadrature is two-dimensional; the numerical matrix itself is generic | Fixed weighted N-body MC through the same matrix/cache and `mean(weights*f)` convention |
| Generation (`kinematics/phase_space_mc.py`, `toy*.py`) | Three daughters, Dalitz proposals and two-dimensional inverse CDF | Recursive `NBodyPhaseSpaceMC`; exact continuous rejection toy in the example |
| Plotting (`plotting.py`, `workflow.py`) | Projection lookup used sample attributes; Dalitz maps use two axes | Generic observable lookup for subset masses and helicity angles; existing 1D projections and projection GOF |
| Fit workflow (`workflow.py`) | Fit core is generic; ROOT constructors, model export, some reports, folding, CP and time-dependent helpers assume `DecayModel` | Existing signal `FitSession.fit()`, acceptance/veto callables, `SignalPDF`, low-level cache and `Minimizer` |

This is additive rather than a rewrite of `DecayModel`/`PhaseSpaceSample`.
Dalitz grids, Square-Dalitz maps, histogram efficiencies/backgrounds defined on
that plane, SCF maps, ROOT/model serialization, CP/time-dependent convenience
sessions, `fit(update_model=True)`, and `FitSession.report/print_fit_fractions`
are **not four-body APIs** in this first version. Use the model/cache fraction
methods directly. Four-body projections do not imply a valid two-dimensional
goodness-of-fit test; mass/angle projections are diagnostics, not complete 5D GOF.

## Minimal construction

```python
from jaxpwa import (
    AmplitudeComponent, CascadeChain, DecayChannel, FitSession,
    FourBodyDecayModel, Isobar, NBodyDecayChannel, NBodyPhaseSpaceMC, PairChain,
    Parameter, RealImag, Resonance,
)

channel = NBodyDecayChannel(2.0, (0.1, 0.2, 0.3, 0.4))
normalization = NBodyPhaseSpaceMC(
    channel.parent_mass, channel.daughter_masses,
).generate(200_000, seed=812)

model = FourBodyDecayModel(channel, [
    AmplitudeComponent("pair", PairChain(
        Isobar(0.65, 0.30), Isobar(0.95, 0.35),
    ), RealImag(1.0, 0.0)),
    AmplitudeComponent("cascade", CascadeChain(
        Isobar(1.45, 0.35), Isobar(0.95, 0.35),
    ), RealImag(
        Parameter.coefficient("cascade.x", 0.25),
        Parameter.coefficient("cascade.y", 0.10),
    )),
], normalization_sample=normalization)

# data: NBodySample containing observed events; keep the reference coefficient fixed.
# session = FitSession(model, data)
# result = session.fit()
# session.plot_projection(result, "s234", projection_sample=normalization)
```

Three-body and N-body declarations expose the same particle-backed factories:

```python
three_body = DecayChannel.from_particles("D+", ("pi-", "pi+", "pi+"))
four_body = NBodyDecayChannel.from_particles(
    "B0", ("K+", "K-", "pi+", "pi-")
)

rho_three_body = Resonance.from_particle(
    "rho(770)0", pair=(0, 1), coefficient=RealImag(1.0, 0.0)
)
rho = Isobar.from_particle("rho(770)0")
# mass=0.77526 GeV, width=0.1474 GeV and spin=1 in the installed database
```

Both channel factories resolve masses and particle identities; the N-body
factory additionally checks that the external spins are zero. `Resonance` and
`Isobar` call the same internal resolver for their nominal mass, width and spin.
The database stores mass and width in MeV; Jax-PWA converts both to GeV.
Explicit `mass=`, `width=`, `spin=`, `lineshape=` and `radius=` values override
the database. Mass, width and radius overrides may be fit `Parameter` objects;
spin must remain a static integer because it fixes the compiled helicity sum.
The original lookup label is retained as `particle_name` for provenance. Database
values depend on the installed `particle` version and should be recorded with an
analysis configuration rather than assumed to be immutable.

For three-body components, `Resonance.from_particle(..., name="rho_12")`
separates the amplitude-component name from the particle-database name. The
legacy `Resonance("rho", ..., mass=..., width=..., spin=...)` construction stays
available for models with explicit values or non-catalogue effective states.

## Fixing parameters after model construction

`DecayModel` and `FourBodyDecayModel` expose the same immutable operation:

```python
# Keep the current values and fix these parameters.
fixed_model = model.with_fixed_parameters("rho.mass", "rho.width")

# Assign a new value and fix it in the returned model.
fixed_model = model.with_fixed_parameters(values={"rho.mass": 0.77526})

# Both forms can be combined.
fixed_model = model.with_fixed_parameters(
    "rho.width", values={"rho.mass": 0.77526}
)
```

Only selected `Parameter` objects change. The original model remains unchanged,
and an unknown name raises an error rather than being silently ignored. External
Monte Carlo normalization samples are reused; compiled caches are rebuilt for
the new free/fixed split. Parameters shared with a `SympyLineshape` remain the
same object in both the resonance/isobar and its symbolic bindings. Build a new
`FitSession` from the returned model.

Numerical construction assumes scalar external particles. Equal masses do not
establish particle identity. Supply `final_state_ids` to enable automatic Bose
symmetrization (or use the particle constructor). The amplitude is the sum over
identity-preserving permutations divided by the square root of their count.
The integration domain remains labelled and unfolded; absolute decay rates
require the conventional identical-particle factorial. `symmetrize=False` is
available for amplitudes already symmetrized by their author.

## Invariant coordinates and frame conventions

`NBodySample.momenta` uses `(E, px, py, pz)` and metric `(+---)`.
Four-vectors are a **covariant storage representation**, not invariant scalars.
`sample.mass_squared(0,2,3)` gives `s134`; `sample.invariants()` gives the
Minkowski Gram matrix and (for four daughters) `det(p_a,p_b,p_c,p_d)`.
The determinant is a pseudoscalar under parity and invariant under proper
Lorentz transformations. Pair masses alone discard this orientation sign.

For `P -> (ab)(cd)`, `pair_coordinates(p)` returns
`s_ab,s_cd,cos_theta1,cos_theta2,phi`. Both helicity polar axes point along their
resonance's flight direction in P. Fix a's azimuth at zero with R1 along +z;
R2 uses axes `(-x,y,-z)`, obtained with `R_y(pi)`. `phi` is c's signed azimuth
in those R2 axes, in `[-pi,pi]`. `pair_coordinates_to_momenta` reconstructs a
canonical rest-frame event from these five invariant coordinates. It preserves
the Gram matrix and the signed orientation; overall spatial orientation is
unobservable for this scalar-parent model. Invalid coordinates return NaNs.

For `P -> R a, R -> S b, S -> c d`, `cascade_coordinates(p)` returns
`s_bcd,s_cd,cos_theta_R,cos_theta_S,phi`. The R polar axis is opposite a in R's
rest frame; S's axes are obtained by the helicity rotation about that axis.
c is boosted **sequentially** through R to S, preserving the inherited axes.
All functions accept an explicit zero-based `order=(a,b,c,d)` permutation.
Angles at threshold/collinear boundaries are conventional (azimuth is set to
zero when undefined); the continuous phase-space measure there vanishes.

`sample.observable("s34")`, `"s234"`, `"cos_theta1"`, `"cos_theta2"` and
`"phi"` work with existing `FitSession.plot_projection` and projection GOF.
The default angular observables use pairing `(01)(23)`; for another pairing
or cascade use the coordinate functions and `plot_binned_data` explicitly.

## Spin, LS couplings and radial factors

Every chain is one LS wave. Distinct allowed orbital momenta are separate
`AmplitudeComponent`s with independent complex coefficients. The Wigner
convention is `D*_(m,n)(phi,theta,0)=exp(+i*m*phi) d_(m,n)(theta)`, with
Condon-Shortley phases. Event-independent normalization factors can be
absorbed in the fitted coefficient; per-component normalization defaults to on.

For the pair topology, scalar-parent angular momentum conservation requires
`L=S`, with `|J1-J2| <= L <= J1+J2`. The implemented factor is

```
sum_h (-1)^(L+J2-h) <J1 h, J2 -h | L 0>
      d^J1_(h,0)(theta1) d^J2_(h,0)(theta2) exp(i*h*phi).
```

The `(-1)^(J2-h)` phase converts the R2 helicity axes into the common-axis
LS basis. Omitting it changes the relative longitudinal/transverse signs.
For `J1=J2=1,L=0` the result is
`(cos(theta1)*cos(theta2) + sin(theta1)*sin(theta2)*cos(phi))/sqrt(3)`;
an independent canonical-vector contraction tests this convention.

For the cascade, production has `L_P=J_R`, the inner decay has `L_S=J_S`, and
the declared `orbital=L_R` must couple with `J_S` to `J_R`. The factor is

```
sum_h sqrt((2*L_R+1)/(2*J_R+1)) <L_R 0, J_S h | J_R h>
      d^J_R_(0,h)(theta_R) d^J_S_(h,0)(theta_S) exp(i*h*phi).
```

Each vertex multiplies by `q^L/sqrt(P_L(q*r))`, reusing the package's **raw**
Blatt-Weisskopf polynomials. These are not its three-body `CovariantAngular`
factors, and no equivalence with those conventions is implied. A plugin's
running width keeps its own pole-normalized barrier prescription. Parity
selection rules are not imposed automatically; choose waves appropriate to
the weak/strong vertices under study.

`Isobar` defaults to the existing fixed-width **`Pole`**, whose convention is
`1/(m-m0-i*Gamma/2)`, not an s-plane Breit-Wigner. Supply
`lineshape=RelativisticBreitWigner()` or another existing lineshape for stable
daughter pairs. For `R -> S b` with unstable S, a two-stable-body RBW width is
not a three-body width: its nominal breakup momentum can vanish when S is
off shell. `CascadeChain` explicitly rejects that choice for its outer
isobar. Use the fixed-width pole or supply a dedicated physical width plugin;
dispersive three-body widths and coupled-channel rescattering are future work.

The helicity-frame conventions follow the issues discussed in
[Marangotto, arXiv:1911.10025](https://arxiv.org/abs/1911.10025); the exact
scalar-external formulas above define this implementation. External spin,
half-integer spin, polarization and Wigner rotations of external spin states
are not implemented.

## Phase space and normalization

`NBodyPhaseSpaceMC` recursively samples the invariant mass squared of the
remaining cluster and an isotropic two-body decay. Every event is on shell
and conserves four-momentum. Each recursion contributes

```
Delta(s)/(2*pi) * q/(4*pi*M)
```

to the proposal weight (the final two-body step has no `Delta(s)`). The
measure is `dPhi_n=(2*pi)^4 delta^4(P-sum(p_i)) product[d^3p_i/((2*pi)^3 2E_i)]`,
so a massless four-body decay integrates to `M^4/(24576*pi^5)`. This is the
same absolute phase-space convention as the existing three-body MC, not the
unscaled `ds12 ds13` convention of the Dalitz quadrature grids. The recursion
follows the [PDG kinematics review](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-kinematics.pdf),
with `(2*pi)^4` absorbed into dPhi here rather than into the decay-rate formula.

Generated events are a **weighted proposal**, not uniformly distributed in
phase space. All integrals use `mean(sample.weights*f)`; do not replace the
weights by one. `with_importance_weights(q)` and `select_for_integration(mask)`
have the same semantics as their three-body counterparts. `weighted_resample`
accepts both sample classes, returns unit-weight pseudo-data, and remains a
finite-pool approximation. The closure example instead generates independent
continuous accept-reject toys using a proven bound for its specific model.

Normalization samples stay fixed during fitting. The same
`PreparedAmplitudeCache` handles coefficients, efficiencies and moving dynamics.
Spin geometry is prepared once, including every identical-particle permutation;
floating mass/width/radius evaluations retain compact geometry arrays. Fixed
dynamics retain only the data component values and the small Hermitian matrix,
and the model reuses that normalization for subsequent data samples.
For floating dynamics declare `Parameter.dynamics(..., owner=component_name)`;
backend names must be unique within each component. JIT and autodiff tests
compare cached/direct intensities, efficiency-weighted normalizations, and
coefficient/mass/width gradients against finite differences.

## Reproduce the closure and tests

```bash
python examples/four_body_closure.py --events 6000 --normalization 200000
pytest tests/test_four_body.py
```

The example uses both scalar topologies coherently, fixes the pair amplitude
coefficient to `1+0i`, fits the cascade coefficient, writes a JSON result and
mass/angle projection PNG, and fails if Minuit is invalid or a pull exceeds 5.
The [notebook](../notebooks/tutorials/tutorial_66_four_body_closure.ipynb) runs
the same example and demonstrates vector-resonance waves.

One validation run (6,000 toy events; 200,000 independent normalization events)
recovered `x=0.547923 +/- 0.025594` and `y=0.359425 +/- 0.024955`, for injected
`x=0.55,y=0.35`, with pulls `-0.081,+0.378`. This is a smoke closure, not a
multi-toy coverage certification. Minuit errors do not include finite-MC
integration uncertainty; test convergence with larger samples and independent
normalization seeds, especially for narrow resonances.

Physical tests cover massive and massless events, near-threshold generation,
absolute massless N-body volumes, massive four-body volume against independent
pair-mass quadrature, permutation/angular moments, proper Lorentz invariance,
orientation under parity, five-coordinate inversion, Wigner orthogonality,
analytic spin limits, Bose symmetry, and the shared fit workflow.
