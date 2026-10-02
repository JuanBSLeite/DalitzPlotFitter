# Toy generation and ROOT output

`generate_signal_toy`, `generate_toy` and `generate_cp_toy` generate unweighted pseudo-data. There are exactly two public sampling methods:

```text
inverse-transform
accept-reject
```

`inverse-transform` is the default because it is dramatically faster for the full amplitude model while retaining continuous, duplicate-free events. `accept-reject` remains available explicitly as an independent reference and validation method.

## Default: inverse transform on the Dalitz plane

The simplest call now uses inverse transform automatically:

```python
toy = generate_toy(
    model,
    100_000,
    parameters=truth,
    seed=2,
)
```

This is equivalent to

```python
toy = generate_toy(
    model,
    100_000,
    parameters=truth,
    method="inverse-transform",
    inverse_resolution=1024,
    seed=2,
)
```

The sampler is a numerical Rosenblatt transform on the Dalitz plane: it tabulates the target density on a rectangular $(m_{12}, v)$ grid, inverts the marginal CDF in $m_{12}$, then the conditional CDF in $v$, with bilinear interpolation of the tabulated quantiles. The full derivation is in [Formulation of the samplers](#formulation-of-the-samplers).

Each candidate is checked against the target density; candidates outside its support are rejected and replaced (with a limit of 100 batches). The generated invariants remain continuous. This support check does not remove the finite-grid approximation inside the allowed region: validate distributions by increasing both CDF resolutions, or compare with `accept-reject`.

> **Narrow resonances in $s_{13}$ or $s_{23}$.** The grid has $N$ points in $v$, and a structure in $s_{13}$ or $s_{23}$ is a *diagonal* band in $(m_{12}, v)$. At the default `inverse_resolution=1024` the $v$ spacing alone is $\sim 0.01$–$0.02\ \mathrm{GeV}^2$ for $B^+\to K^+K^+K^-$, larger than the $\phi(1020)$ full width at half maximum ($\approx 0.009\ \mathrm{GeV}^2$), so the peak is broadened and shifted. Use `method="accept-reject"` for such models; see [Narrow structures](#narrow-structures-measured-bias-and-remedies).

Four-momenta are reconstructed in the parent rest frame from the sampled invariants and then given a random global orientation, so ROOT toy output retains the same momentum branches as accept-reject generation.

Preparation and generation run entirely in JAX: the CDF tables are built in one JIT-compiled program (all conditional rows at once via `vmap`), candidates are drawn with `jax.random` from the integer `seed`, and the momentum reconstruction is the same JAX routine used by `PhaseSpaceMC.attach_momenta`. Only two scalars per batch (density validity and the count of in-support candidates) reach the host. The first call in a process pays a one-time JIT compilation cost; repeated `prepared.generate(...)` calls with the same size reuse it, which is where a toy campaign spends its time. The same `seed` reproduces the same toy, but samples are not event-by-event identical to those produced before the sampler moved from `numpy.random` to `jax.random`.

`inverse_resolution` controls the number of grid points in both Dalitz directions. `inverse_quantile_resolution` can be supplied separately for the tabulated conditional inverse CDF; by default it equals `inverse_resolution`.

## Reuse the prepared inverse CDFs

The expensive part of inverse-transform generation is preparation. For repeated pseudoexperiments with the same model parameters, prepare once:

```python
from jaxpwa import prepare_inverse_toy_generator

prepared = prepare_inverse_toy_generator(
    model,
    parameters=truth,
    efficiency=efficiency,
    veto=veto,
    resolution=1024,
)

toy1 = prepared.generate(100_000, seed=1)
toy2 = prepared.generate(100_000, seed=2)
toy3 = prepared.generate(1_000_000, seed=3)
```

The CDF grids are evaluated once during preparation. Subsequent `generate` calls reuse them and evaluate the target density on candidate events to verify support, before four-momentum reconstruction and optional component shuffling. Keep the supplied model and density callbacks unchanged while reusing a prepared generator.

If model parameters change, the prepared generator must be rebuilt because the target CDFs change.

## Accept-reject reference sampler

```python
toy = generate_toy(
    model,
    100_000,
    parameters=truth,
    method="accept-reject",
    seed=1,
)
```

The accept-reject implementation follows the Laura++ safety principle: if a proposal exceeds its current envelope, already accepted events from that component are discarded and generation restarts with an enlarged envelope. Probabilities are never clipped.

For invariant-only densities the proposal square is automatically divided into equal Dalitz cells. A pilot sample estimates one envelope per cell; cells are then drawn with probability proportional to their local envelope and proposals inside a selected cell are uniform. Because all cells have equal area, accepting with `score / local_envelope` reproduces the same exact target density while avoiding the very poor efficiency of a single global maximum. The cell grid is occupancy-aware and grows up to 32x32 for large pilots. Momentum-dependent custom densities retain the monitored global-envelope fallback.

The accept-reject proposal retains the exact weighted `PhaseSpaceMC` measure, but its invariant-only path is optimized for rejection sampling. Candidate pools are generated directly in `s12/s13/s23` without constructing four-momenta. Component normalization scales and coefficients are frozen once at the requested toy truth, and coefficient-only models reuse the same fixed-normalization template already cached by the fitter. By default the stratified proposal batch uses the same array size as the pilot, so JAX does not compile the full amplitude once for the pilot shape and a second time for a larger proposal shape. The proposal density is JIT-compiled, and acceptance decisions stay on the JAX device apart from the compact boolean selection mask.

If four-momenta are requested, they are reconstructed only for the final accepted events. Rejected candidates therefore never pay the cost of parent-rest-frame orientations and boosts. A custom efficiency, veto, or amplitude that explicitly requests `p1/p2/p3` automatically falls back to the full proposal representation. Accepted four-momenta are retained so their angular selection is preserved. In mixtures, only compact components have moments reconstructed; `include_momenta=False` discards moments after selection.

The sampler intentionally keeps a monitored global envelope as an independent validation path. That global envelope can still be inefficient for strongly structured amplitude models; proposal-shape improvements are a separate optimization from the exact computational fast path described above.

## Square-Dalitz histogram backgrounds

For a histogram density `h(mprime, thetaprime)` per Square-Dalitz area, use
`SquareDalitzHistogramBackground(..., divide_jacobian=True)`. Both
`ToyBackground` and `CPToyBackground`, including `with_charge_asymmetry()`
wrappers, then sample in the histogram's ordered Square-Dalitz coordinate pair:

- `accept-reject` proposes uniformly on the full unit square and accepts using
  the raw histogram height, with monitored envelopes and restart on overflow.
- `inverse-transform` tabulates its marginal and conditional CDFs on that square;
  `prepare_inverse_toy_generator` reuses these tables as usual.

Vetoes are evaluated on the corresponding invariants. Folded histograms are
sampled on the full square, looking up the folded angle, so both exchange-related
halves are populated. The fit and CP charge-split integrals still evaluate `h/J`
with ordinary Dalitz integration weights. The same post-veto normalization and
charge scales therefore determine generation and fitting.

Earlier versions used raw `h` with the ordinary Dalitz sampling measure,
producing `h*J` in Square Dalitz and potentially incorrect CP charge fractions.
Merely substituting `h/J` in a conventional-Dalitz sampler is numerically unsafe
at the boundaries, where `J` vanishes. Direct square-coordinate sampling avoids
that singularity. Inverse-CDF endpoint values use inward limits at 1e-6 from
the square edges, where the helicity angle remains defined.

With `divide_jacobian=False`, the histogram callable is instead a density per
ordinary Dalitz area and generation preserves that interpretation. Custom
callables also supply density per ordinary Dalitz area; a `generation_value`
attribute alone no longer changes their measure.

## Performance

For data/model plots, `CPFitSession.prepare_projection_toy(result, method=...)`
generates reusable per-charge signal/background toys. Pass the returned
`CPProjectionToy` to `session.plot_projection_from_toy` for the same bins,
folding, selection, axes and pull options as `plot_projection`, without
regenerating events for every plot. See [generated CP projections](user_friendly_api.md#cp-projections-from-generated-toys).
These samples retain fitted charge/component yields and acceptance; they must
not be used as unit-weight normalization samples.

The repository benchmark should be used for current timings because both the accept-reject and compact phase-space paths are actively optimized. It reports accept-reject separately with and without retained four-momenta, as well as inverse-transform preparation and prepared-generation throughput.

## Formulation of the samplers

This section states exactly what each sampler computes, so that its accuracy limits can be judged from the equations rather than discovered empirically. Source: `toy_inverse.py` and `inverse_transform.py` (inverse transform), `toy_accept.py` and `kinematics/phase_space_mc.py` (accept-reject).

### Common setup

Let $M$ be the parent mass and $m_1,m_2,m_3$ the daughter masses. The Dalitz plane is parameterized by $(s_{12},s_{13})$, with

$$
s_{23}=M^2+m_1^2+m_2^2+m_3^2-s_{12}-s_{13},\qquad
s_{\min}=(m_1+m_2)^2\le s_{12}\le s_{\max}=(M-m_3)^2 .
$$

At fixed $s_{12}$ the invariant $s_{13}$ is confined to $[s_{13}^-(s_{12}),\,s_{13}^+(s_{12})]$ with width

$$
W(s_{12})=s_{13}^+(s_{12})-s_{13}^-(s_{12}).
$$

Every method draws unweighted events from a normalized density proportional to a *target density per ordinary Dalitz area* $f(s_{12},s_{13})$:

$$
p(s_{12},s_{13})=\frac{f(s_{12},s_{13})}{\displaystyle\int_{\mathcal D} f\,\mathrm ds_{12}\,\mathrm ds_{13}},
\qquad
f=\epsilon\,V\,|\mathcal A|^2 \quad\text{(signal)},
$$

where $\epsilon$ is the efficiency, $V$ the veto (0 or 1) and $\mathcal A$ the coherent amplitude at the fixed toy parameters. A background component uses its own shape times the veto in place of $\epsilon V|\mathcal A|^2$. The per-component normalization is irrelevant for sampling: only the shape of $f$ matters.

**Mixtures.** For a total of $N$ events with signal fraction $f_s$, the signal count is $N_s\sim\mathrm{Binomial}(N,f_s)$ and the background counts are multinomial with the fitted background weights. For a CP toy, the signal events are split between charges as

$$
N_+\sim\mathrm{Binomial}\!\left(N_s,\ \frac{I_+}{I_++I_-}\right),\qquad
I_q=\int_{\mathcal D}\epsilon_q V_q\,|\mathcal A_q|^2\,\mathrm ds_{12}\,\mathrm ds_{13},
$$

which is the same joint-normalization convention as `CPJointNLL`. Each component is then sampled independently and the merged sample is shuffled.

**Momenta.** Both samplers return only invariants. Four-momenta, when requested, are reconstructed afterwards in the parent rest frame from $(s_{12},s_{13},s_{23})$ with a uniformly random global orientation.

### Inverse transform (Rosenblatt)

**Change of variables.** Use $m=\sqrt{s_{12}}$ and the fractional position $v$ inside the $s_{13}$ range:

$$
s_{12}=m^2,\qquad s_{13}=s_{13}^-(s_{12})+v\,W(s_{12}),\qquad 0\le v\le 1 .
$$

The Jacobian is $\mathrm ds_{12}\,\mathrm ds_{13}=2m\,W(m^2)\,\mathrm dm\,\mathrm dv$, so the joint density in $(m,v)$ is

$$
g(m,v)=2m\,W(m^2)\,f\big(m^2,\ s_{13}^-+vW\big).
$$

**Rosenblatt transform.** Define the marginal and conditional distributions

$$
p_M(m)=\int_0^1 g(m,v)\,\mathrm dv,\qquad
F_M(m)=\frac{\int_{m_{\min}}^{m}p_M}{\int p_M},\qquad
F_{V|M}(v\mid m)=\frac{\int_0^v g(m,v')\,\mathrm dv'}{\int_0^1 g(m,v')\,\mathrm dv'} .
$$

If $U_1,U_2$ are independent uniform variables on $[0,1]$, then

$$
m=F_M^{-1}(U_1),\qquad v=F_{V|M}^{-1}(U_2\mid m)
$$

is distributed exactly as $g/\!\int g$. All approximation in the implementation comes from tabulating these two inverses on finite grids.

**Tabulation.** With resolution $N$ (`inverse_resolution`) and $N_q$ quantile levels (`inverse_quantile_resolution`, default $N$):

- a uniform grid $m_i$, $i=0,\dots,N-1$, on $[m_1+m_2,\ M-m_3]$ (the two endpoints are moved one floating-point step inward, where $W=0$);
- a uniform grid $v_k$, $k=0,\dots,N-1$, on $[0,1]$;
- the density $g_{ik}=g(m_i,v_k)$, evaluated for every grid point in one call to the target density.

Row-wise cumulative trapezoids give $C_{ik}\approx\int_0^{v_k}g(m_i,v')\,\mathrm dv'$. The row totals $P_i=C_{i,N-1}$ give the marginal $p_M(m_i)$, and a second cumulative trapezoid over $m_i$ gives the tabulated $F_M(m_i)$. For each row $i$ and each level $p_j=j/(N_q-1)$ the quantile table is

$$
Q_{ij}=F_{V|M}^{-1}(p_j\mid m_i),
$$

obtained by linear interpolation of the normalized row CDF $C_{ik}/P_i$. A plateau of the CDF (zero density) is a *jump* of the inverse rather than a segment to interpolate across, so forbidden intervals are never populated.

**Generation.** For each event draw $U_1,U_2$ and:

1. $m=F_M^{-1}(U_1)$ by linear interpolation of the tabulated $F_M$;
2. locate the row $i$ with $m_i\le m<m_{i+1}$ and set $\lambda=(m-m_i)/(m_{i+1}-m_i)$;
3. write $U_2(N_q-1)=j+\tau$ with $j$ an integer and $0\le\tau<1$, and interpolate the quantile tables bilinearly:

$$
v=(1-\lambda)\big[(1-\tau)Q_{ij}+\tau Q_{i,j+1}\big]+\lambda\big[(1-\tau)Q_{i+1,j}+\tau Q_{i+1,j+1}\big];
$$

4. set $s_{12}=m^2$, $s_{13}=s_{13}^-+vW$ and $s_{23}$ from the sum rule.

Each candidate is then re-evaluated with the *exact* density. Candidates with $f=0$ (outside the support of $\epsilon V|\mathcal A|^2$, for example inside a veto) are discarded and replaced, for at most 100 batches. This removes forbidden regions but does not correct the density inside the allowed region.

**Square-Dalitz variant.** For histogram backgrounds defined on the Square Dalitz plane, the same algorithm runs directly on $(m',\theta')\in[0,1]^2$ with the callback supplying the density per $\mathrm dm'\,\mathrm d\theta'$. The Jacobian factor is then $1$ (not $2mW$), and $m'$ takes the role of $m$.

**Error sources.** There are three, all controlled by $N$ and $N_q$:

1. the trapezoidal integration along $v$;
2. the linear inversion of the tabulated CDFs;
3. the interpolation of the quantile tables *between neighboring rows* $m_i$, $m_{i+1}$. This interpolates quantile functions, not densities: if the location of a narrow structure in $v$ differs between the two rows, the interpolated quantile function describes a smeared structure.

Error 3 is the dominant one for a narrow resonance in $s_{13}$ or $s_{23}$. Its band is $s_{13}\approx m_R^2$, that is

$$
v_{\rm band}(m)=\frac{m_R^2-s_{13}^-(m^2)}{W(m^2)},
$$

which depends on $m$: the band is diagonal in the $(m,v)$ grid. A resonance of mass $m_R$ and width $\Gamma_R$ has a full width at half maximum of $2m_R\Gamma_R$ in $s_{13}$. Resolving it requires both

$$
\frac{W(s_{12})}{N-1}\ll 2m_R\Gamma_R
\qquad\text{and}\qquad
\left|\frac{\partial s_{13}^{\rm band}}{\partial m}\right|\frac{m_{\max}-m_{\min}}{N-1}\ll 2m_R\Gamma_R ,
$$

where $s_{13}^{\rm band}=s_{13}^-+v_{\rm band}W$. Both conditions get $N$-fold harder to meet for a narrower resonance, and the cost of raising $N$ is quadratic: the tables have $N\times N$ (and $N\times N_q$) entries. Numbers for $B^+\to K^+K^+K^-$ with $\phi(1020)$ ($2m_R\Gamma_R\approx 0.0087\ \mathrm{GeV}^2$) and $N=1024$ are given in [Narrow structures](#narrow-structures-measured-bias-and-remedies).

### Accept-reject (monitored local envelope)

**Proposal and score.** Proposals are uniform on the unit square $(u,v)\in[0,1]^2$, with

$$
s_{12}=s_{\min}+(s_{\max}-s_{\min})\,u,\qquad s_{13}=s_{13}^-(s_{12})+v\,W(s_{12}).
$$

This is the invariant projection of the weighted `PhaseSpaceMC` measure. Its Jacobian is $\mathrm ds_{12}\,\mathrm ds_{13}=(s_{\max}-s_{\min})\,W(s_{12})\,\mathrm du\,\mathrm dv$, and each proposal carries the phase-space weight

$$
w(s_{12})=\frac{(s_{\max}-s_{\min})\,W(s_{12})}{128\pi^3M^2}.
$$

The *score* of a proposal is $\sigma=w\,f$. Because $w$ is exactly the Jacobian to $(u,v)$ up to a constant, $\sigma$ is proportional to the target density per unit area of the $(u,v)$ square,

$$
\sigma(u,v)\ \propto\ f\,\frac{\partial(s_{12},s_{13})}{\partial(u,v)},
$$

so uniform proposals in $(u,v)$ followed by accept-reject on $\sigma$ sample $p$ exactly. Candidate pools are generated directly in $(s_{12},s_{13},s_{23})$; four-momenta are never built for rejected candidates.

**Local envelopes.** The unit square is divided into $n_u\times n_v$ equal cells $c$, with

$$
n_u=n_v=\min\!\Big(32,\ \max\!\big(2,\ \lfloor\sqrt{n_{\rm pilot}/32}\rfloor\big)\Big),
$$

where $n_{\rm pilot}$ is the pilot size (`pool_size`). A pilot sample of $n_{\rm pilot}$ uniform proposals gives one envelope per cell,

$$
E_c=\kappa\,\max_{\text{pilot events in }c}\sigma ,\qquad \kappa=\texttt{envelope\_safety}>1 .
$$

A cell that contains no pilot event takes the global pilot maximum, so that no physical region has zero proposal probability. A single global maximum is avoided on purpose: for a structured amplitude it gives a very low efficiency.

**Sampling.** A cell is drawn with probability $P_c=E_c/\sum_{c'}E_{c'}$ and the point is uniform inside the cell. Hence the proposal density per unit $(u,v)$ area is $q=n_un_v\,P_c$. A candidate is accepted with probability $\sigma/E_c$, so the accepted density is

$$
q\cdot\frac{\sigma}{E_c}=\frac{n_un_v}{\sum_{c'}E_{c'}}\,\sigma\ \propto\ \sigma ,
$$

independent of the cell: the cell structure changes only the efficiency, not the target. The expected efficiency is

$$
\varepsilon\simeq\frac{\langle\sigma\rangle}{\langle E\rangle},
$$

estimated on the pilot; it sizes the batches only in the global-envelope fallback below, since the compact path reuses the pilot size. A larger $\kappa$ costs efficiency in proportion to $1/\kappa$.

**Exactness and the monitor.** The result is exact only if $\sigma\le E_c$ for every point. Probabilities are never clipped. In each batch the sampler checks whether any proposal has $\sigma>E_{c}$; if so it

1. raises the offending envelopes, $E_c\leftarrow\max\big(E_c,\ \kappa\max_{\text{batch in }c}\sigma\big)$,
2. discards all events accepted so far, since events accepted under a too-low envelope under-represent the region where it was exceeded, and
3. restarts, counting one restart (`max_restarts`, default 10).

When the limit is exceeded the sampler stops with *accept-reject local envelope was exceeded too many times*. The check is statistical: a region with $\sigma>E_c$ is detected with a probability that grows with the number of proposals landing there, and an exceedance never proposed cannot be seen. The bias from such an undetected region is limited to the density mass that proposals there would have carried, which is small when the region is rarely proposed, but it is not bounded by a guarantee.

**Pilot occupancy for narrow peaks.** For an isolated Breit-Wigner peak the density falls from its maximum as

$$
\frac{|\mathrm{BW}(\delta)|^2}{|\mathrm{BW}(0)|^2}=\frac{1}{1+\delta^2/(m_R\Gamma_R)^2},
\qquad \delta=s_{13}-m_R^2 .
$$

The cell envelope is safe only if some pilot event lands within the window where this ratio is $\ge1/\kappa$, that is $|\delta|\le m_R\Gamma_R\sqrt{\kappa-1}$, of full width

$$
\Delta_{\rm win}=2\,m_R\Gamma_R\sqrt{\kappa-1}.
$$

If a cell spans $L=W/n_v$ in $s_{13}$ and contains $n_{\rm pilot}/(n_un_v)$ pilot events, the expected number of pilot events in the window is about

$$
\bar n_{\rm hit}\simeq\frac{n_{\rm pilot}}{n_un_v}\,\frac{\Delta_{\rm win}}{L}.
$$

When $\bar n_{\rm hit}\lesssim 1$ the envelope is usually exceeded and every restart discards the accepted events. This is an order-of-magnitude rule for one isolated peak, not a bound; interference and neighboring resonances change the local shape. Raising `pool_size` increases $\bar n_{\rm hit}$ linearly. Raising `envelope_safety` increases $\Delta_{\rm win}$ as $\sqrt{\kappa-1}$ at the price of efficiency. Raising `max_restarts` helps only once the envelope is nearly right, because each restart converges the envelope only where a proposal exceeded it.

**Fallbacks.** Densities that request four-momentum fields, and Square-Dalitz histogram backgrounds, use a *global* envelope $E=\kappa\max\sigma$ over the pilot, with the same monitored restart. For Square-Dalitz backgrounds the proposal is uniform on the full $(m',\theta')$ square and $\sigma$ is the raw histogram height, which avoids the $1/J$ singularity at the Dalitz boundary.

**Parameters.** `pool_size` is the pilot size $n_{\rm pilot}$ (default $\max(20\,000,\min(100\,000,2N))$ for $N$ requested events). It affects only the envelopes, never the accepted events. `batch_size` is the number of candidates per proposal round (default: the pilot size for the compact proposal, so that JAX compiles the amplitude once); it affects only speed and device memory. `envelope_safety` is $\kappa$ (default 1.20) and `max_restarts` the restart limit (default 10).

### Choosing between them

| | inverse-transform | accept-reject |
|---|---|---|
| Exact in the limit | $N,N_q\to\infty$ (interpolated tables) | exact up to the monitored envelope |
| Dominant error | table interpolation across rows and along $v$ | undetected envelope exceedance |
| Narrow structure in $s_{13}$/$s_{23}$ | broadened and shifted unless $N$ is very large | correct if the pilot samples the peak |
| Cost | $O(N^2)$ preparation, $O(1)$ per event | $\propto 1/\varepsilon$ per event, restarts waste events |
| Reuse for repeated toys | prepared once, `prepared.generate(...)` | no preparation, no reuse |
| Typical use | smooth models, large toy campaigns | narrow resonances, validation, projections |

## Accuracy and validation

Inverse-transform sampling is interpolated rather than analytically exact. Its numerical accuracy is controlled by the preparation resolution and should be validated against deterministic projections and the accept-reject reference sampler.

The repository benchmark compares the two methods using the full B+ -> K+ pi+ pi- model:

```bash
python benchmarks/benchmark_toy_generation.py --size 100000
```

The benchmark reports:

- compact accept-reject end-to-end time and proposal efficiency;
- accept-reject time when final four-momenta are requested;
- inverse-transform preparation + first-generation time;
- generation time from an already prepared inverse sampler;
- 1D projection closure and 2D Dalitz total-variation distance.

The benchmark workflow is manual (`workflow_dispatch`) because the 1,000,000-event jobs are intentionally expensive.

### Narrow structures: measured bias and remedies

The inverse-transform error analysis above predicts a bias for a narrow resonance in $s_{13}$ or $s_{23}$. For $B^+\to K^+K^+K^-$ with a single $\phi(1020)$ ($m_R=1.0195\ \mathrm{GeV}$, $\Gamma_R=4.25\ \mathrm{MeV}$, so $2m_R\Gamma_R\approx0.0087\ \mathrm{GeV}^2$) and $N=1024$, the grid quantities are:

| $m_{12}$ (GeV) | $W$ ($\mathrm{GeV}^2$) | $W/(N-1)$ ($\mathrm{GeV}^2$) |
|---|---|---|
| 1.2 | 14.9 | 0.0145 |
| 2.0 | 20.5 | 0.0200 |
| 3.0 | 17.4 | 0.0170 |
| 4.0 | 10.6 | 0.0104 |

At $m_{12}=2\ \mathrm{GeV}$ the band also moves by about $0.007\ \mathrm{GeV}^2$ in $s_{13}$ between adjacent rows ($\Delta m_{12}=3.7\ \mathrm{MeV}$). Both numbers are comparable to or larger than the resonance width, so neither resolution condition holds. Generating $200\,000$ events and looking at $s_{\rm low}=\min(s_{13},s_{23})\in[0.98,1.10]\ \mathrm{GeV}^2$ gives:

| method | median $s_{\rm low}$ ($\mathrm{GeV}^2$) | central 68% width ($\mathrm{GeV}^2$) |
|---|---|---|
| inverse-transform, $N=1024$ | 1.0419 | 0.0230 |
| inverse-transform, $N=2048$ | 1.0406 | 0.0176 |
| accept-reject | 1.0400 | 0.0153 |

The inverse-transform peak is shifted and broadened, and converges toward accept-reject as $N$ grows. $N=4096$ needs several $N\times N$ tables and ran out of memory on a 4 GB GPU. The single-resonance numbers are illustrative, not a general tolerance: repeat the check for your own model.

For accept-reject the corresponding failure is different: with the default $100\,000$-event pilot, the same model repeatedly exceeds its local envelope and stops with *accept-reject local envelope was exceeded too many times*. In this model $n_u=n_v=32$ and $W/n_v\approx0.64\ \mathrm{GeV}^2$ at $m_{12}=2\ \mathrm{GeV}$; with $\kappa=1.2$ the window is $\Delta_{\rm win}\approx0.0039\ \mathrm{GeV}^2$, so $\bar n_{\rm hit}\approx(100\,000/1024)\times(0.0039/0.64)\approx0.6$ for a $100\,000$-event pilot and $\approx6$ for $1\,000\,000$. This matches what was observed: $500\,000$ events failed with `pool_size=100_000` for both seeds tried and succeeded with `pool_size=1_000_000`, in 6 to 13 seconds.

Recommended practice for models with narrow resonances:

- use `method="accept-reject"` with `pool_size` large enough that $\bar n_{\rm hit}\gtrsim5$; the pilot is evaluated once, so a larger pool costs memory and one pass, not per-event time;
- keep `batch_size` at or below what fits on the device; it does not change the result;
- raise `envelope_safety` or `max_restarts` only as a secondary measure;
- when `inverse-transform` is kept for speed, compare a projection onto the narrow variable against `accept-reject` before trusting it.

## Method-specific options

`pool_size`, `batch_size`, `envelope_safety` and `max_restarts` belong to `accept-reject`.

`inverse_resolution` and `inverse_quantile_resolution` belong to `inverse-transform`.

`pool_size` is the accept-reject *pilot* size: it only sets the envelopes, never the accepted events, and must be large enough to sample narrow peaks. `batch_size` is the number of candidates per proposal round and affects only speed and memory. Definitions and the occupancy rule are in [Formulation of the samplers](#formulation-of-the-samplers).

Passing `pool_size` or `batch_size` together with the default inverse-transform method is rejected rather than silently ignored. If those options are needed, set `method="accept-reject"` explicitly.

## Compact toys for memory-constrained fits

By default generated toys retain reconstructed four-momenta together with the
three Dalitz invariants. If the downstream fit, plotting, efficiency and veto
models use only `s12`, `s13` and `s23`, the four-momenta can be omitted:

```python
toy = generate_toy(
    model,
    1_000_000,
    parameters=truth,
    seed=2,
    include_momenta=False,
)
```

For float64 arrays this reduces the retained array payload for one million
unweighted events from about 128 MiB to about 32 MiB. In both toy-generation paths, `include_momenta=False` skips final momentum
reconstruction, reducing peak memory and generation work. Accept-reject
candidate pools are invariant-only even when final momenta are requested, so
rejected candidates never construct four-vectors. The compact and full
phase-space generators use independent random-coordinate streams; a common
seed guarantees reproducibility within each mode, not event-by-event identity
between the two representations.

Existing samples can be compacted with

```python
compact = sample.without_momenta()
print(compact.nbytes)
```

The default remains `include_momenta=True` for backward compatibility and for
workflows that need momentum branches in ROOT output.

## Save a non-CP toy to ROOT

```python
from jaxpwa import generate_toy

toy = generate_toy(
    model,
    100_000,
    parameters=truth,
    seed=1,
    output_root="toy.root",
)
```

The default tree is `DecayTree`. It contains `s12`, `s13`, `s23`, `weight`, and, when available in the generated `PhaseSpaceSample`, the four-momentum branches

```text
p1_E  p1_PX  p1_PY  p1_PZ
p2_E  p2_PX  p2_PY  p2_PZ
p3_E  p3_PX  p3_PY  p3_PZ
```

The tree name can be changed with `output_tree=`. The generated sample is still returned in memory.

## Save a CP toy to one ROOT tree

```python
plus_toy, minus_toy = generate_cp_toy(
    plus_model,
    minus_model,
    100_000,
    parameters=truth,
    seed=2,
    output_root="cp_toy.root",
)
```

The signal and background charge splits use the same deterministic accepted-integral convention in both sampling paths. Both charges are written to the same `DecayTree`. A signed integer branch named `charge` identifies the sample:

```text
charge = +1  -> B+
charge = -1  -> B-
```

This makes charge selections straightforward in uproot or ROOT while retaining one common file/tree:

```python
import uproot

with uproot.open("cp_toy.root") as f:
    tree = f["DecayTree"]
    plus = tree.arrays(cut="charge > 0", library="np")
    minus = tree.arrays(cut="charge < 0", library="np")
```

ROOT output is implemented with `uproot`; PyROOT is not required.

Components with zero generation weight are not prepared. CP generation prepares only charges with positive event counts, permits one charge to have zero rate, and requires finite non-negative charge integrals with a positive sum for each sampled component. Pure-background toys do not evaluate the absent signal normalization.
