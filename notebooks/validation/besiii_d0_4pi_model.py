"""D0 -> pi+ pi- pi+ pi- isobar model after BESIII, arXiv:2312.02524 (Tables 5-11).

Built on the helicity/LS four-body API (``jaxpwa.four_body``); the paper uses the
covariant Zemach (Rarita-Schwinger) tensor formalism, so complex couplings are
NOT transferable. Only fit fractions / interference fractions are compared
(see the notebook). Daughter order everywhere: (pi+_0, pi-_1, pi+_2, pi-_3).

Waves (all 14 groups of Table 11, 17 components):
  a1(1260)+-pi-+  = rho pi[S] + rho pi[D] + f2 pi[P] + (pipi)_S pi[P]   (Table 9)
  a1(1420)+pi-    -> f0(980) pi [P]
  a1(1640)+-pi-+  -> rho pi [S]
  a2(1320)+-pi-+  -> rho pi [D]
  pi(1300)+-pi-+  = rho pi[P] + (pipi)_S pi[S]                          (Table 9)
  rho0 rho0 [S,P,D], rho0 rho(1450)0 [P,D], rho0 (pipi)_S, f2(1270) (pipi)_S,
  (pipi)_S (pipi)_S (2D P-vector, Eqs. 19-20)

a1(1260) and pi(1300) use the running width of Eq. 21: Gamma(s) is obtained by
integrating |A(rho pi)|^2 over the pi pi pi phase space (dominant sub-decay only;
the other sub-decays and the pi+pi0pi0 channel are neglected in Gamma(s) and the
result is normalised to Gamma0 at the pole). Other 3-pi resonances use a
constant width, as in the paper.

Deviations (documented in the notebook): helicity/LS instead of Zemach tensors;
raw Blatt-Weisskopf barriers of the package; PDG-like inputs for the parameters
the paper does not tabulate (a1(1420), a1(1640), a2(1320), rho(1450), f2 P-vector).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from jaxpwa import (
    AmplitudeComponent,
    CascadeChain,
    Isobar,
    NBodyDecayChannel,
    NBodyPhaseSpaceMC,
    NBodySample,
    PairChain,
)
from jaxpwa.dynamics.lineshape.flatte import Flatte
from jaxpwa.dynamics.lineshape.gounaris_sakurai import GounarisSakurai
from jaxpwa.dynamics.lineshape.kmatrix import (
    _POLE_COUPLINGS,
    _POLE_MASSES,
    KMatrix,
    _slowly_varying_factor,
)
from jaxpwa.dynamics.lineshape.pole import Pole
from jaxpwa.dynamics.lineshape.relativistic_breit_wigner import (
    RelativisticBreitWigner,
)
from jaxpwa.dynamics.sequential import _radial
from jaxpwa.kinematics.vectors import invariant_mass_squared

M_D0, M_PI = 1.86484, 0.13957039
CHANNEL = NBodyDecayChannel(M_D0, (M_PI,) * 4, (211, -211, 211, -211))
R_RES = 3.0  # GeV^-1, intermediate resonances (paper Sec. 5)
R_D = 5.0  # GeV^-1, D0

# Table 10 (this work) for a1(1260), pi(1300); PDG-like otherwise.
A1 = dict(mass=1.193, width=0.487)
PI1300 = dict(mass=1.534, width=0.610)
A1420 = dict(mass=1.411, width=0.158)
A1640 = dict(mass=1.655, width=0.254)
A2 = dict(mass=1.3182, width=0.1054)
RHO = dict(mass=0.77526, width=0.1491)
RHO1450 = dict(mass=1.465, width=0.400)
F2 = dict(mass=1.2755, width=0.1867)


def cplx(mag, phase):
    return complex(mag * np.cos(phase), mag * np.sin(phase))


def rho_iso():
    return Isobar(RHO["mass"], RHO["width"], 1, GounarisSakurai(), R_RES)


def rho1450_iso():
    return Isobar(RHO1450["mass"], RHO1450["width"], 1, GounarisSakurai(), R_RES)


def f2_iso():
    return Isobar(F2["mass"], F2["width"], 2, RelativisticBreitWigner(), R_RES)


def f0_980_iso():
    return Isobar(0.965, 0.1, 0, Flatte.f0_980(), R_RES)


def s_iso(betas=(1.0,), f_pipi=0j, f_kk=0j):
    """(pipi)_S from the Anisovich-Sarantsev K-matrix + P-vector (paper Eq. 16-18)."""
    b = tuple(betas) + (0j,) * (5 - len(betas))
    return Isobar(0.5, 0.5, 0, KMatrix(betas=b, f_prod=(f_pipi, f_kk, 0j, 0j, 0j)), R_RES)


# P-vectors (Tables 8 and 9).
S_RHO_S = dict(betas=(cplx(8.4, -1.68),), f_pipi=cplx(40.7, -0.50), f_kk=cplx(121, 1.73))
S_F2_S = dict(betas=(1.0 + 0j,), f_pipi=cplx(18.3, -1.39), f_kk=cplx(56, 2.29))
S_IN_A1 = dict(betas=(cplx(0.83, -2.18),), f_pipi=cplx(2.47, 0.34), f_kk=cplx(6.59, 1.76))
S_IN_PI1300 = dict(betas=(cplx(5.0, -0.64),), f_pipi=0j, f_kk=cplx(43.8, 0.34))


# ------------------------------------------------- running width, Eq. 21
def _three_pion_g(mass, kind, n, seed):
    """g(m) = int sum_spins |A(R -> rho pi)|^2 dPhi3 at resonance mass ``mass``.

    kind "a1": J=1, S-wave, spin-summed -V.V* + |V.P|^2/M^2 with V = sum_pairings
    f_pair r_pair; kind "pi": J=0, P-wave, A = sum_pairings f_pair (r . p_bachelor).
    f_pair = GS(m_rho) * q^1 B_1 (rho decay) * q^L B_L (R decay, raw barriers).
    """
    events = NBodyPhaseSpaceMC(mass, (M_PI,) * 3).generate(n, seed=seed)
    p, w = events.momenta, events.weights  # 0: pi+_a, 1: pi+_b, 2: pi-
    metric = jnp.array([1.0, -1.0, -1.0, -1.0])
    dot = lambda a, b: jnp.sum(a * b * metric, axis=-1)
    total = p.sum(axis=1)
    orbital = 0 if kind == "a1" else 1
    v = 0
    for plus, bachelor in ((0, 1), (1, 0)):
        pair, r = p[:, plus] + p[:, 2], p[:, plus] - p[:, 2]
        m_rho = jnp.sqrt(jnp.maximum(dot(pair, pair), 0.0))
        f = rho_iso().evaluate(m_rho, M_PI, M_PI, mass, M_PI, 1, None)
        f = f * _radial(mass, m_rho, M_PI, orbital, R_RES)
        v = v + (f[:, None] * r if kind == "a1" else f * dot(r, p[:, bachelor]))
    if kind == "a1":
        amp2 = -jnp.sum(jnp.conj(v) * v * metric, axis=-1).real
        amp2 = amp2 + jnp.abs(jnp.sum(v * total * metric, axis=-1)) ** 2 / mass**2
    else:
        amp2 = jnp.abs(v) ** 2
    return float(jnp.mean(w * amp2))


@lru_cache(maxsize=None)
def running_width_table(kind, points=90, events=300_000):
    """Smoothed g(m) on a mass grid from 3 m_pi to the D0 - m_pi endpoint."""
    from scipy.signal import savgol_filter

    grid = np.linspace(3 * M_PI + 2e-3, M_D0 - M_PI, points)
    g = np.array([_three_pion_g(m, kind, events, seed=i) for i, m in enumerate(grid)])
    return grid, np.maximum(savgol_filter(g, 9, 3), 0.0)


@dataclass(frozen=True, eq=False)
class RunningWidthPole:
    """1/(m0^2 - s - i m0 Gamma0 g(s)/g(m0^2)), paper Eqs. 14 and 21 (numerical g)."""

    kind: str = "a1"

    def __post_init__(self):
        running_width_table(self.kind)  # fill the cache eagerly, outside any jit trace

    def __call__(self, mass, context):
        grid, table = running_width_table(self.kind)
        m0, gamma0 = context.pole_mass, context.pole_width
        g = jnp.interp(mass, grid, table, left=0.0)
        g0 = jnp.interp(m0, grid, table)
        return 1.0 / (m0**2 - mass**2 - 1j * m0 * gamma0 * g / g0)


def _outer(params, spin, lineshape=None):
    return Isobar(params["mass"], params["width"], spin,
                  Pole() if lineshape is None else lineshape, R_RES)


# Charge assignments: X+ pi- has bachelor pi-_1 and X -> rho0 pi+_0; X- pi+ swaps them.
ORDER_PLUS = (1, 0, 2, 3)
ORDER_MINUS = (0, 1, 2, 3)


def _cascade(outer, inner, orbital, order):
    return CascadeChain(outer, inner, orbital, order, R_D)


@dataclass(frozen=True)
class CoherentSum:
    """U = sum_k ratio_k * U_k with FIXED complex relative couplings (paper Table 9).

    The sub-decay pattern of a resonance is shared by its charge states, so it is
    one component; ``ratios[0]`` is the reference sub-decay (1).
    """

    chains: tuple
    ratios: tuple

    def prepare_data(self, data):
        for chain in self.chains:
            if hasattr(chain, "prepare_data"):
                data = chain.prepare_data(data)
        return data

    def compact_prepared_data(self, data):
        out = {}
        for chain in self.chains:
            if hasattr(chain, "compact_prepared_data"):
                out.update(chain.compact_prepared_data(data))
            else:  # covariant chains work directly from the four-momenta
                out["momenta"] = data["momenta"]
        return out

    def __call__(self, data, parameters=None):
        return sum(r * chain(data, parameters) for r, chain in zip(self.ratios, self.chains))


def a1_chains(order):
    rho_s = _cascade(_outer(A1, 1, RunningWidthPole("a1")), rho_iso(), 0, order)
    rho_d = _cascade(_outer(A1, 1, RunningWidthPole("a1")), rho_iso(), 2, order)
    f2_p = _cascade(_outer(A1, 1, RunningWidthPole("a1")), f2_iso(), 1, order)
    s_p = _cascade(_outer(A1, 1, RunningWidthPole("a1")), s_iso(**S_IN_A1), 1, order)
    return (rho_s, rho_d, f2_p, s_p)


def pi1300_chains(order):
    rho_p = _cascade(_outer(PI1300, 0, RunningWidthPole("pi")), rho_iso(), 1, order)
    s_s = _cascade(_outer(PI1300, 0, RunningWidthPole("pi")), s_iso(**S_IN_PI1300), 0, order)
    return (rho_p, s_s)


# Table 9 (charge +-1): relative FFs (%) and phases of the sub-decays vs rho pi.
A1_SUB_FF = (79.7, 1.3, 1.8, 5.4)  # rho pi[S], rho pi[D], f2 pi[P], (pipi)_S pi[P]
A1_SUB_PHASE = (0.0, 0.01, -1.58, 0.0)
PI1300_SUB_FF = (53.8, 51.1)
PI1300_SUB_PHASE = (0.0, 0.0)


def other_chain(name):
    if name == "a1420p":
        return _cascade(_outer(A1420, 1), f0_980_iso(), 1, ORDER_PLUS)
    if name in ("a1640p", "a1640m"):
        return _cascade(_outer(A1640, 1), rho_iso(), 0, ORDER_PLUS if name[-1] == "p" else ORDER_MINUS)
    if name in ("a2p", "a2m"):
        return _cascade(_outer(A2, 2), rho_iso(), 2, ORDER_PLUS if name[-1] == "p" else ORDER_MINUS)
    raise KeyError(name)


def rho_rho(orbital):
    return PairChain(rho_iso(), rho_iso(), orbital, (0, 1, 2, 3), R_D)


def rho_rho1450(orbital):
    return PairChain(rho_iso(), rho1450_iso(), orbital, (0, 1, 2, 3), R_D)


def rho_s():
    return PairChain(rho_iso(), s_iso(**S_RHO_S), 1, (0, 1, 2, 3), R_D)


def f2_s():
    return PairChain(f2_iso(), s_iso(**S_F2_S), 2, (0, 1, 2, 3), R_D)


def _pair_masses_sq(p):
    return invariant_mass_squared(p[:, 0] + p[:, 1]), invariant_mass_squared(
        p[:, 2] + p[:, 3]
    )


@dataclass(frozen=True)
class DoubleSWave:
    """(pipi)_S (pipi)_S with the 2D P-vector of paper Eqs. 19-20.

    F(s1,s2) = sum_{rho,sigma} R_rho(s1) R_sigma(s2) P_{rho sigma}(s1,s2), with
    R_rho = [(I - i K rho)^-1]_{0 rho} and rho, sigma in {pipi, KK}. Parameter
    reading (the text is ambiguous; see notebook): only a_{1,1}, a_{1,2},
    b_{2,pipi}, c[pipi,pipi], c[pipi,KK] are non-zero (Table 8); c[pipi,KK]
    enters P_{pipi,KK} only (upper triangle). Pole masses/couplings are the
    package's Anisovich-Sarantsev K-matrix, the same set as the paper's Ref. [45].
    """

    a11: complex = cplx(2224, -1.044)
    a12: complex = cplx(7287, 1.727)
    b2_pipi: complex = cplx(8816, -1.107)
    c_pipi: complex = cplx(2433, 1.796)
    c_pikk: complex = cplx(5417, 2.68)
    s0: float = -5.0

    def _response(self, s):
        return KMatrix().prepare_mass(jnp.sqrt(s))

    def _pvector(self, s1, s2):
        """P_{rho sigma}(s1, s2) over all five K-matrix channels (Eq. 20).

        The pole terms must run over every channel: only then does the residue
        of P at s = m_alpha^2 stay proportional to the K-matrix coupling vector
        g^alpha, which is what cancels the pole against (I - iK rho)^-1.
        """
        g = _POLE_COUPLINGS[:2, :]  # g[alpha, channel], poles 1-2, five channels
        m2 = _POLE_MASSES[:2] ** 2
        sv1 = _slowly_varying_factor(s1, self.s0)[..., None, None]
        sv2 = _slowly_varying_factor(s2, self.s0)[..., None, None]
        d1 = 1.0 / (m2 - s1[..., None])  # (events, alpha)
        d2 = 1.0 / (m2 - s2[..., None])
        total = jnp.zeros(s1.shape + (5, 5), dtype=complex)
        for (al, be), coef in {(0, 0): self.a11, (0, 1): self.a12}.items():
            total += coef * (
                g[al][:, None] * g[be][None, :] * (d1[..., al] * d2[..., be])[..., None, None]
                + g[be][:, None] * g[al][None, :] * (d1[..., be] * d2[..., al])[..., None, None]
            )
        # b_{2,pipi}: channel index 0 in whichever S-wave carries the slow term.
        al = 1
        e0 = jnp.zeros(5).at[0].set(1.0)
        total += self.b2_pipi * sv1 * (e0[:, None] * g[al][None, :]) * d2[..., al][..., None, None]
        total += self.b2_pipi * sv2 * (g[al][:, None] * e0[None, :]) * d1[..., al][..., None, None]
        cmat = jnp.zeros((5, 5), dtype=complex)
        cmat = cmat.at[0, 0].set(self.c_pipi).at[0, 1].set(self.c_pikk)
        total += cmat * (sv1 * sv2)
        return total

    def __call__(self, data, parameters=None):
        s1, s2 = _pair_masses_sq(data["momenta"])
        r1, r2 = self._response(s1), self._response(s2)
        return jnp.einsum("...r,...s,...rs->...", r1, r2, self._pvector(s1, s2))



# ------------------------------------------------------------ tables and model
# name: (label, group, Table 8 FF %, stat err %, Table 8 phase [rad])
WAVES = {
    "a1p": ("a1(1260)+ pi-", "a1p", 82.2, 3.3, 0.0),
    "a1m": ("a1(1260)- pi+", "a1m", 10.3, 1.5, 0.23),
    "a1420p": ("a1(1420)+ pi-", "a1420p", 0.6, 0.2, 2.70),
    "a1640p": ("a1(1640)+ pi-", "a1640p", 1.7, 0.5, -2.07),
    "a1640m": ("a1(1640)- pi+", "a1640m", 0.5, 0.3, -1.26),
    "a2p": ("a2(1320)+ pi-", "a2p", 0.2, 0.1, -2.92),
    "a2m": ("a2(1320)- pi+", "a2m", 0.3, 0.1, -0.47),
    "pi1300p": ("pi(1300)+ pi-", "pi1300p", 32.3, 2.6, -2.325),
    "pi1300m": ("pi(1300)- pi+", "pi1300m", 23.5, 2.3, -2.631),
    "rhorho_S": ("rho0 rho0 [S]", "rhorho", 1.7, 0.6, -3.10),
    "rhorho_P": ("rho0 rho0 [P]", "rhorho", 9.8, 1.0, 1.62),
    "rhorho_D": ("rho0 rho0 [D]", "rhorho", 23.1, 2.1, -3.06),
    "rhorho1450_P": ("rho0 rho(1450)0 [P]", "rhorho1450", 1.0, 0.4, 0.68),
    "rhorho1450_D": ("rho0 rho(1450)0 [D]", "rhorho1450", 1.5, 0.9, 3.08),
    "rho_S": ("rho0 (pipi)_S", "rho_S", 2.7, 0.6, 0.0),
    "SS": ("(pipi)_S (pipi)_S", "SS", 62.8, 4.6, 0.0),
    "f2_S": ("f2(1270) (pipi)_S", "f2_S", 1.8, 0.4, 0.0),
}
NAMES = tuple(WAVES)
GROUPS = ("a1p", "a1m", "a1420p", "a1640p", "a1640m", "a2p", "a2m", "pi1300p",
          "pi1300m", "rhorho", "rhorho1450", "rho_S", "SS", "f2_S")
GROUP_FF = {  # Table 8/11 diagonal: (FF %, stat err %)
    "a1p": (82.2, 3.3), "a1m": (10.3, 1.5), "a1420p": (0.6, 0.2), "a1640p": (1.7, 0.5),
    "a1640m": (0.5, 0.3), "a2p": (0.2, 0.1), "a2m": (0.3, 0.1), "pi1300p": (32.3, 2.6),
    "pi1300m": (23.5, 2.3), "rhorho": (28.0, 1.9), "rhorho1450": (2.5, 0.9),
    "rho_S": (2.7, 0.6), "SS": (62.8, 4.6), "f2_S": (1.8, 0.4),
}
# Table 11 (pi+pi-pi+pi-), upper triangle in GROUPS order: row -> values for the
# following columns, each (value %, stat err %).
_T11_ROWS = {
    "a1p": [(9.9, .8), (.5, .3), (-15.3, 2.5), (-.4, .7), (0, 0), (0, 0), (1.7, .2), (-27.3, 2.1),
            (-4.6, 1.9), (-5.2, 1.5), (-1.3, 2.3), (5.2, 1.1), (-1.4, .2)],
    "a1m": [(-.1, .1), (-1.4, .3), (-2.5, .8), (0, 0), (0, 0), (-10.8, 1.2), (-.1, .1),
            (-1.8, .7), (-1.5, .5), (-.3, .8), (3.4, .5), (-.4, .1)],
    "a1420p": [(.1, .1), (0, 0), (0, 0), (0, 0), (-.2, 0), (.1, .1), (-.3, .1), (-.1, .1),
               (-.2, .1), (1.0, .2), (-.2, 0)],
    "a1640p": [(.2, .1), (0, 0), (0, 0), (.2, .1), (4.9, .7), (-1.1, .3), (.3, .1), (.1, .3),
               (-1.6, .6), (.1, 0)],
    "a1640m": [(0, 0), (0, 0), (2.7, .8), (.3, .1), (-.2, .2), (.1, .1), (.4, .3), (-1.9, .5), (0, 0)],
    "a2p": [(-.1, 0), (0, 0), (0, 0), (-.8, .3), (.2, .1), (0, 0), (0, 0), (0, 0)],
    "a2m": [(0, 0), (0, 0), (1.0, .2), (-.2, .1), (0, 0), (0, 0), (0, 0)],
    "pi1300p": [(9.2, 1.5), (-9.3, .7), (-2.8, .8), (4.0, 1.6), (-49.2, 4.1), (.3, .3)],
    "pi1300m": [(-6.0, .7), (-2.6, .7), (-2.8, 1.5), (-39.8, 3.9), (-.1, .3)],
    "rhorho": [(0, 2.1), (0, 0), (2.0, .5), (-1.2, .2)],
    "rhorho1450": [(0, 0), (-.6, .3), (-.3, .1)],
    "rho_S": [(0, 0), (0, 0)],
    "SS": [(-1.3, .4)],
}
TABLE11 = {
    (g, GROUPS[GROUPS.index(g) + 1 + k]): v
    for g, row in _T11_ROWS.items()
    for k, v in enumerate(row)
}
assert len(TABLE11) == 91


def paper_targets():
    """Sum of all Table 8 diagonal + Table 11 interference terms (= 100 % up to rounding)."""
    return sum(v for v, _ in GROUP_FF.values()) + sum(v for v, _ in TABLE11.values())


def subdecay_ratios(norm, model_cls=None):
    """Complex relative couplings of the sub-decays from Table 9 (relative FFs, phases).

    |ratio_k|^2 = (FF_k / FF_0) * (raw int|U_0|^2 / raw int|U_k|^2); phases are the
    paper's Table 9 phases (0 for the (pipi)_S pi parts, whose phase sits in the P-vector).
    """
    from jaxpwa import FourBodyDecayModel, RealImag

    out = {}
    families = {
        "a1": (a1_chains(ORDER_PLUS), A1_SUB_FF, A1_SUB_PHASE),
        "pi1300": (pi1300_chains(ORDER_PLUS), PI1300_SUB_FF, PI1300_SUB_PHASE),
    }
    for family, (chains, ff, phase) in families.items():
        model = FourBodyDecayModel(
            CHANNEL,
            [AmplitudeComponent(f"sub{k}", ch, RealImag(1.0, 0.0)) for k, ch in enumerate(chains)],
            normalization_sample=norm, normalize_components=False)
        raw = np.diag(hermitian_matrix(component_matrix(model, norm), norm.weights)).real
        out[family] = tuple(
            complex(np.sqrt(ff[k] / ff[0] * raw[0] / raw[k]) * np.exp(1j * phase[k]))
            for k in range(len(chains))
        )
    return out


def build_components(ratios, coefficients, covariant=False):
    """AmplitudeComponent list; ``coefficients`` maps name -> RealImag-like object.

    ``covariant=True`` switches to the paper's Zemach spin factors (``zemach_components``).
    """
    if covariant:
        return zemach_components(coefficients, ratios)
    funcs = {
        "a1p": CoherentSum(a1_chains(ORDER_PLUS), ratios["a1"]),
        "a1m": CoherentSum(a1_chains(ORDER_MINUS), ratios["a1"]),
        "a1420p": other_chain("a1420p"),
        "a1640p": other_chain("a1640p"),
        "a1640m": other_chain("a1640m"),
        "a2p": other_chain("a2p"),
        "a2m": other_chain("a2m"),
        "pi1300p": CoherentSum(pi1300_chains(ORDER_PLUS), ratios["pi1300"]),
        "pi1300m": CoherentSum(pi1300_chains(ORDER_MINUS), ratios["pi1300"]),
        "rhorho_S": rho_rho(0),
        "rhorho_P": rho_rho(1),
        "rhorho_D": rho_rho(2),
        "rhorho1450_P": rho_rho1450(1),
        "rhorho1450_D": rho_rho1450(2),
        "rho_S": rho_s(),
        "SS": DoubleSWave(),
        "f2_S": f2_s(),
    }
    return [AmplitudeComponent(n, funcs[n], coefficients[n]) for n in NAMES]


def cp_conjugate(sample: NBodySample) -> NBodySample:
    """CP image: swap pi+ <-> pi- labels and reverse all three-momenta."""
    p = sample.momenta[:, jnp.array([1, 0, 3, 2]), :]
    p = p.at[..., 1:].multiply(-1.0)
    return NBodySample(p, sample.weights)


def symmetric_union(sample: NBodySample) -> NBodySample:
    bar = cp_conjugate(sample)
    return NBodySample(
        jnp.concatenate([sample.momenta, bar.momenta]),
        jnp.concatenate([sample.weights, bar.weights]),
    )


# ---------------------------------------------------------------- calibration
_G = np.zeros((len(NAMES), len(GROUPS)))
for _i, _n in enumerate(NAMES):
    _G[_i, GROUPS.index(WAVES[_n][1])] = 1.0


def fractions_from_matrix(c, M):
    """FF_i, merged-group FF_g (incl. intra-group interference) and pair fractions.

    FF_i = |c_i|^2 M_ii / T, T = c^dagger M c; the pair fraction of groups (g, h) is
    2 Re(sum_{i in g, j in h} conj(c_i) c_j M_ij) / T, as in Eqs. 40-41.
    """
    c = np.asarray(c, dtype=complex)
    weight = np.conj(c)[:, None] * c[None, :] * M
    total = weight.sum().real
    diag = {n: weight[i, i].real / total for i, n in enumerate(NAMES)}
    blocks = _G.T @ weight @ _G
    gdiag = {g: blocks[k, k].real / total for k, g in enumerate(GROUPS)}
    pairs = {(g, h): 2 * blocks[GROUPS.index(g), GROUPS.index(h)].real / total for (g, h) in TABLE11}
    return diag, gdiag, pairs


def calibrate(M, seed=0, restarts=40):
    """Fit the 16 free complex coefficients (a1+ fixed = 1) to Tables 8 and 11.

    Residuals (units of the paper's statistical error, floored at 0.3 %): 17
    component FFs, 14 group FFs (rho rho includes S/P/D interference) and 91
    Table 11 pair fractions. Multi-start least squares, first start from Table 8.
    """
    from scipy.optimize import least_squares

    n = len(NAMES) - 1
    ff_target = np.array([WAVES[k][2] for k in NAMES])
    ff_err = np.maximum([WAVES[k][3] for k in NAMES], 0.3)
    g_target = np.array([GROUP_FF[g][0] for g in GROUPS])
    g_err = np.maximum([GROUP_FF[g][1] for g in GROUPS], 0.3)
    p_keys = list(TABLE11)
    p_target = np.array([TABLE11[k][0] for k in p_keys])
    p_err = np.maximum([TABLE11[k][1] for k in p_keys], 0.3)

    def unpack(x):
        c = np.ones(len(NAMES), dtype=complex)
        c[1:] = x[:n] * np.exp(1j * x[n:])
        return c

    def residual(x):
        d, gd, p = fractions_from_matrix(unpack(x), M)
        return np.concatenate([
            (100 * np.array([d[k] for k in NAMES]) - ff_target) / ff_err,
            (100 * np.array([gd[g] for g in GROUPS]) - g_target) / g_err,
            (100 * np.array([p[k] for k in p_keys]) - p_target) / p_err,
        ])

    rng = np.random.default_rng(seed)
    phase0 = np.array([WAVES[k][4] for k in NAMES[1:]])
    mag_ref = np.sqrt(np.array([WAVES[k][2] for k in NAMES[1:]]) / WAVES["a1p"][2])
    lower = np.concatenate([np.full(n, 1e-4), np.full(n, -2 * np.pi)])
    upper = np.concatenate([np.full(n, 50.0), np.full(n, 2 * np.pi)])
    best = None
    for k in range(restarts):
        mag0 = mag_ref * (np.exp(0.3 * rng.standard_normal(n)) if k else 1.0)
        ph0 = phase0 if k == 0 else rng.uniform(-np.pi, np.pi, n)
        sol = least_squares(residual, np.concatenate([mag0, ph0]), bounds=(lower, upper))
        if best is None or sol.cost < best.cost:
            best = sol
    return unpack(best.x), 2 * best.cost, len(best.fun)


# ------------------------------------------------ covariant (Zemach) formalism
@dataclass(frozen=True)
class ConstWidthBW:
    """1/(m0^2 - s - i m0 Gamma0): relativistic BW with constant width (paper Eq. 14)."""

    def __call__(self, mass, context):
        return 1.0 / (context.pole_mass**2 - mass**2 - 1j * context.pole_mass * context.pole_width)


# Paper couplings, Table 8 (magnitude, phase) and Table 9 (sub-decay ratios).
PAPER_LAMBDA = {
    "a1p": cplx(100.0, 0.0), "a1m": cplx(35.3, 0.23), "a1420p": cplx(19.0, 2.70),
    "a1640p": cplx(20.1, -2.07), "a1640m": cplx(10.5, -1.26), "a2p": cplx(0.23, -2.92),
    "a2m": cplx(0.30, -0.47), "pi1300p": cplx(76.3, -2.325), "pi1300m": cplx(65.1, -2.631),
    "rhorho_S": cplx(6.1, -3.10), "rhorho_P": cplx(6.17, 1.62), "rhorho_D": cplx(4.54, -3.06),
    "rhorho1450_P": cplx(13.9, 0.68), "rhorho1450_D": cplx(5.6, 3.08),
    "rho_S": 1.0 + 0j, "SS": 1.0 + 0j, "f2_S": 1.0 + 0j,
}
PAPER_RATIOS = {
    "a1": (1.0 + 0j, cplx(0.060, 0.01), cplx(0.311, -1.58), 1.0 + 0j),
    "pi1300": (1.0 + 0j, 1.0 + 0j),
}


def _iso(par, spin, lineshape):
    return Isobar(par["mass"], par["width"], spin, lineshape, R_RES)


def a1_chains_z(order):
    from besiii_zemach import ZCascade

    a1 = lambda: _iso(A1, 1, RunningWidthPole("a1"))
    return (
        ZCascade("a_rhoS", a1(), rho_iso(), 1, 0, order),
        ZCascade("a_rhoD", a1(), rho_iso(), 1, 2, order),
        ZCascade("a_fP", a1(), f2_iso(), 1, 1, order),
        ZCascade("a_SP", a1(), s_iso(**S_IN_A1), 1, 1, order),
    )


def pi_chains_z(order):
    from besiii_zemach import ZCascade

    pi1300 = lambda: _iso(PI1300, 0, RunningWidthPole("pi"))
    return (
        ZCascade("pi_rho", pi1300(), rho_iso(), 0, 1, order),
        ZCascade("pi_S", pi1300(), s_iso(**S_IN_PI1300), 0, 0, order),
    )


def zemach_single(name):
    """Covariant chain for every component that is a single chain."""
    from besiii_zemach import ZCascade, ZPair

    bw = ConstWidthBW()
    return {
        "a1420p": lambda: ZCascade("a_SP", _iso(A1420, 1, bw), f0_980_iso(), 1, 1, ORDER_PLUS),
        "a1640p": lambda: ZCascade("a_rhoS", _iso(A1640, 1, bw), rho_iso(), 1, 0, ORDER_PLUS),
        "a1640m": lambda: ZCascade("a_rhoS", _iso(A1640, 1, bw), rho_iso(), 1, 0, ORDER_MINUS),
        "a2p": lambda: ZCascade("a2_rhoD", _iso(A2, 2, bw), rho_iso(), 2, 2, ORDER_PLUS),
        "a2m": lambda: ZCascade("a2_rhoD", _iso(A2, 2, bw), rho_iso(), 2, 2, ORDER_MINUS),
        "rhorho_S": lambda: ZPair("VV_S", rho_iso(), rho_iso(), 0),
        "rhorho_P": lambda: ZPair("VV_P", rho_iso(), rho_iso(), 1),
        "rhorho_D": lambda: ZPair("VV_D", rho_iso(), rho_iso(), 2),
        "rhorho1450_P": lambda: ZPair("VV_P", rho_iso(), rho1450_iso(), 1),
        "rhorho1450_D": lambda: ZPair("VV_D", rho_iso(), rho1450_iso(), 2),
        "rho_S": lambda: ZPair("VS_P", rho_iso(), s_iso(**S_RHO_S), 1),
        "SS": lambda: DoubleSWave(),
        "f2_S": lambda: ZPair("TS_D", f2_iso(), s_iso(**S_F2_S), 2),
    }[name]()


# ZPair waves with identical isobars (and the symmetric double S-wave) are summed over
# the 2 distinct Bose permutations in the paper, not the package's 4: halve their amplitude.
BOSE_DUPLICATED = ("rhorho_S", "rhorho_P", "rhorho_D", "SS")


def zemach_components(coefficients, ratios=None):
    """Covariant-tensor version of ``build_components`` (same names and order).

    Spin factors, barriers and the Bose-symmetrisation follow paper Eqs. 6-10 and
    Table 5; lineshapes are the paper's (GS for rho, Eq. 21 running widths for
    a1(1260)/pi(1300), constant-width BW for the other 3-pi resonances).
    """
    ratios = PAPER_RATIOS if ratios is None else ratios
    funcs = {
        "a1p": CoherentSum(a1_chains_z(ORDER_PLUS), ratios["a1"]),
        "a1m": CoherentSum(a1_chains_z(ORDER_MINUS), ratios["a1"]),
        "pi1300p": CoherentSum(pi_chains_z(ORDER_PLUS), ratios["pi1300"]),
        "pi1300m": CoherentSum(pi_chains_z(ORDER_MINUS), ratios["pi1300"]),
    }
    return [
        AmplitudeComponent(n, funcs[n] if n in funcs else zemach_single(n), coefficients[n])
        for n in NAMES
    ]


def zemach_basis_functions():
    """The 25 raw covariant chains in BASIS order (see ``basis_functions``)."""
    chains = {"a1p": a1_chains_z(ORDER_PLUS), "a1m": a1_chains_z(ORDER_MINUS),
              "pi1300p": pi_chains_z(ORDER_PLUS), "pi1300m": pi_chains_z(ORDER_MINUS)}
    return [chains[n][k] if n in chains else zemach_single(n) for n, k in BASIS]


# ------------------------------------------ extended-basis calibration (Table 9)
# 25 raw chains; component i = sum_k rho_ik * chain_k with rho shared inside a family.
BASIS = (
    [("a1p", k) for k in range(4)] + [("a1m", k) for k in range(4)]
    + [(n, 0) for n in ("a1420p", "a1640p", "a1640m", "a2p", "a2m")]
    + [("pi1300p", k) for k in range(2)] + [("pi1300m", k) for k in range(2)]
    + [(n, 0) for n in ("rhorho_S", "rhorho_P", "rhorho_D", "rhorho1450_P",
                        "rhorho1450_D", "rho_S", "SS", "f2_S")]
)
_FAMILY = {"a1p": "a1", "a1m": "a1", "pi1300p": "pi1300", "pi1300m": "pi1300"}
# Table 9 relative FFs (%) and stat. errors of the sub-decays of a1(1260), pi(1300).
TABLE9 = {"a1": ((79.7, 2.2), (1.3, 0.3), (1.8, 0.4), (5.4, 0.6)),
          "pi1300": ((53.8, 2.9), (51.1, 2.9))}


def basis_functions():
    """Raw (un-normalised) chains in BASIS order."""
    chains = {"a1p": a1_chains(ORDER_PLUS), "a1m": a1_chains(ORDER_MINUS),
              "pi1300p": pi1300_chains(ORDER_PLUS), "pi1300m": pi1300_chains(ORDER_MINUS)}
    funcs = {"rhorho_S": rho_rho(0), "rhorho_P": rho_rho(1), "rhorho_D": rho_rho(2),
             "rhorho1450_P": rho_rho1450(1), "rhorho1450_D": rho_rho1450(2),
             "rho_S": rho_s(), "SS": DoubleSWave(), "f2_S": f2_s()}
    out = []
    for name, k in BASIS:
        if name in chains:
            out.append(chains[name][k])
        elif name in funcs:
            out.append(funcs[name])
        else:
            out.append(other_chain(name))
    return out


def basis_gram(norm, covariant=False):
    """Hermitian Gram matrix G_kl = mean(w conj(b_k) b_l) of the 25 raw Bose-symmetrised chains."""
    from jaxpwa import FourBodyDecayModel, RealImag

    functions = zemach_basis_functions() if covariant else basis_functions()
    model = FourBodyDecayModel(
        CHANNEL,
        [AmplitudeComponent(f"b{k}", f, RealImag(1.0, 0.0)) for k, f in enumerate(functions)],
        normalization_sample=norm, normalize_components=False)
    return hermitian_matrix(component_matrix(model, norm), norm.weights)


def rho_matrix(ratios):
    """(17, 25) complex map from chains to components; ratios shared within a family."""
    R = np.zeros((len(NAMES), len(BASIS)), dtype=complex)
    for i, name in enumerate(NAMES):
        for k, (bname, sub) in enumerate(BASIS):
            if bname == name:
                R[i, k] = ratios[_FAMILY[name]][sub] if name in _FAMILY else 1.0
    return R


def raw_matrix(G, ratios):
    """Un-normalised Hermitian matrix H_ij = int conj(A_i) A_j of the 17 components."""
    R = rho_matrix(ratios)
    return np.conj(R) @ G @ R.T


def normalized_matrix(G, ratios):
    """Unit-diagonal Hermitian normalisation matrix of the 17 components."""
    R = rho_matrix(ratios)
    H = np.conj(R) @ G @ R.T
    d = np.sqrt(np.diag(H).real)
    return H / d[:, None] / d[None, :]


def family_fractions(G, ratios):
    """Relative sub-decay FFs (%) of each family: |r_k|^2 G_kk / r^dagger G r."""
    out = {}
    for fam, name in (("a1", "a1p"), ("pi1300", "pi1300p")):
        idx = [k for k, (b, _) in enumerate(BASIS) if b == name]
        r = np.asarray(ratios[fam])
        Gf = G[np.ix_(idx, idx)]
        total = (np.conj(r) @ Gf @ r).real
        out[fam] = 100 * np.abs(r) ** 2 * np.diag(Gf).real / total
    return out


def calibrate_full(G, ratios0, seed=0, restarts=30):
    """Fit couplings AND the complex sub-decay ratios to Tables 8, 9 and 11.

    Free: 16 component couplings (a1+ fixed to 1), 3 complex a1 ratios (rho pi[D],
    f2 pi[P], (pipi)_S pi[P]) and 1 complex pi(1300) ratio ((pipi)_S pi[S]); the
    reference sub-decay (rho pi) keeps ratio 1. Residual terms: Table 8 component
    and group FFs, the 91 Table 11 pair fractions and the six Table 9 relative FFs.
    """
    from scipy.optimize import least_squares

    n = len(NAMES) - 1
    ff_t = np.array([WAVES[k][2] for k in NAMES]); ff_e = np.maximum([WAVES[k][3] for k in NAMES], 0.3)
    g_t = np.array([GROUP_FF[g][0] for g in GROUPS]); g_e = np.maximum([GROUP_FF[g][1] for g in GROUPS], 0.3)
    keys = list(TABLE11)
    p_t = np.array([TABLE11[k][0] for k in keys]); p_e = np.maximum([TABLE11[k][1] for k in keys], 0.3)
    t9_t = np.array([v for f in ("a1", "pi1300") for v, _ in TABLE9[f]])
    t9_e = np.array([e for f in ("a1", "pi1300") for _, e in TABLE9[f]])

    def unpack(x):
        c = np.ones(len(NAMES), dtype=complex)
        c[1:] = x[:n] * np.exp(1j * x[n : 2 * n])
        rmag, rph = x[2 * n : 2 * n + 4], x[2 * n + 4 : 2 * n + 8]
        z = rmag * np.exp(1j * rph)
        ratios = {"a1": (1.0 + 0j, z[0], z[1], z[2]), "pi1300": (1.0 + 0j, z[3])}
        return c, ratios

    def residual(x):
        c, ratios = unpack(x)
        d, gd, p = fractions_from_matrix(c, normalized_matrix(G, ratios))
        fam = family_fractions(G, ratios)
        return np.concatenate([
            (100 * np.array([d[k] for k in NAMES]) - ff_t) / ff_e,
            (100 * np.array([gd[g] for g in GROUPS]) - g_t) / g_e,
            (100 * np.array([p[k] for k in keys]) - p_t) / p_e,
            (np.concatenate([fam["a1"], fam["pi1300"]]) - t9_t) / t9_e,
        ])

    rng = np.random.default_rng(seed)
    phase0 = np.array([WAVES[k][4] for k in NAMES[1:]])
    mag_ref = np.sqrt(np.array([WAVES[k][2] for k in NAMES[1:]]) / WAVES["a1p"][2])
    r0 = np.array([abs(ratios0["a1"][1]), abs(ratios0["a1"][2]), abs(ratios0["a1"][3]),
                   abs(ratios0["pi1300"][1])])
    rp0 = np.array([np.angle(ratios0["a1"][1]), np.angle(ratios0["a1"][2]),
                    np.angle(ratios0["a1"][3]), np.angle(ratios0["pi1300"][1])])
    lo = np.concatenate([np.full(n, 1e-4), np.full(n, -2 * np.pi), np.full(4, 1e-4), np.full(4, -2 * np.pi)])
    hi = np.concatenate([np.full(n, 50.0), np.full(n, 2 * np.pi), np.full(4, 1e3), np.full(4, 2 * np.pi)])
    best = None
    for k in range(restarts):
        mag0 = mag_ref * (np.exp(0.3 * rng.standard_normal(n)) if k else 1.0)
        ph0 = phase0 if k == 0 else rng.uniform(-np.pi, np.pi, n)
        rm = r0 * (np.exp(0.2 * rng.standard_normal(4)) if k else 1.0)
        rph = rp0 if k == 0 else rng.uniform(-np.pi, np.pi, 4)
        sol = least_squares(residual, np.concatenate([mag0, ph0, rm, rph]), bounds=(lo, hi))
        if best is None or sol.cost < best.cost:
            best = sol
    c, ratios = unpack(best.x)
    return c, ratios, 2 * best.cost, len(best.fun)


# ------------------------------------ paper couplings with discrete conventions
# Total (stat (+) exp. syst. (+) model syst., quadrature) uncertainties of Table 8 / 9:
# name: (sigma|Lambda|, sigma_phase). Waves without an entry are not constrained.
LAMBDA_ERR = {
    "a1m": (5.4, 0.17), "a1420p": (5.5, 1.11), "a1640p": (7.0, 0.32), "a1640m": (4.7, 0.63),
    "a2p": (0.09, 0.40), "a2m": (0.06, 0.27), "pi1300p": (7.4, 0.30), "pi1300m": (6.4, 0.23),
    "rhorho_S": (1.8, 0.54), "rhorho_P": (0.69, 0.12), "rhorho_D": (0.41, 0.21),
    "rhorho1450_P": (3.0, 0.27), "rhorho1450_D": (1.8, 0.47),
}
RATIO_ERR = {"a1_D": (0.024, 0.21), "a1_f2": (0.043, 0.28)}  # Table 9, rho pi[D], f2 pi[P]


def bose_weights():
    """Amplitude factor 1/2 for identical-isobar waves (2 distinct Bose permutations)."""
    return np.array([0.5 if n in BOSE_DUPLICATED else 1.0 for n in NAMES])


def paper_coupling_fractions(G, shifts, rshifts):
    """Fractions for the paper's magnitudes/phases + discrete convention shifts.

    ``shifts`` (17) are per-component phase offsets and ``rshifts`` (4) offsets of
    the Table 9 ratios (rho pi[D], f2 pi[P], pi(1300) (pipi)_S pi, index 3 for pi).
    """
    c = np.array([PAPER_LAMBDA[n] for n in NAMES]) * bose_weights() * np.exp(1j * np.asarray(shifts))
    base = np.array(PAPER_RATIOS["a1"]) * np.exp(1j * np.r_[0.0, rshifts[:3]])
    ratios = {"a1": tuple(base), "pi1300": tuple(np.array(PAPER_RATIOS["pi1300"]) * np.exp(1j * np.r_[0.0, rshifts[3]]))}
    return c, ratios, fractions_from_matrix(c, raw_matrix(G, ratios))


def _table_residual(d, gd, p):
    ff_t = np.array([WAVES[k][2] for k in NAMES]); ff_e = np.maximum([WAVES[k][3] for k in NAMES], 0.3)
    g_t = np.array([GROUP_FF[g][0] for g in GROUPS]); g_e = np.maximum([GROUP_FF[g][1] for g in GROUPS], 0.3)
    keys = list(TABLE11)
    p_t = np.array([TABLE11[k][0] for k in keys]); p_e = np.maximum([TABLE11[k][1] for k in keys], 0.3)
    return np.concatenate([
        (100 * np.array([d[k] for k in NAMES]) - ff_t) / ff_e,
        (100 * np.array([gd[g] for g in GROUPS]) - g_t) / g_e,
        (100 * np.array([p[k] for k in keys]) - p_t) / p_e,
    ])


def scan_conventions(G, restarts=200, seed=0):
    """Discrete search of per-wave phase conventions (multiples of pi/2).

    The paper does not fix the sign / i of each spin structure (order of r, epsilon
    sign, charge-partner ordering), so the published phases are defined only up to
    such conventions. Coordinate descent over {0, pi/2, pi, 3pi/2} with the paper's
    magnitudes; Table 8/11 residuals as objective. No continuous parameter is fitted.
    """
    def chi2(sh, rs):
        *_, (d, gd, p) = paper_coupling_fractions(G, sh, rs)
        r = _table_residual(d, gd, p)
        return float(r @ r)

    rng = np.random.default_rng(seed)
    n = len(NAMES)
    best = None
    for k in range(restarts):
        sh = rng.integers(0, 4, n) * np.pi / 2 if k else np.zeros(n)
        rs = rng.integers(0, 4, 4) * np.pi / 2 if k else np.zeros(4)
        cur, improved = chi2(sh, rs), True
        while improved:
            improved = False
            for i in range(1, n + 4):
                for step in (1, 2, 3):
                    t, tr = sh.copy(), rs.copy()
                    if i < n:
                        t[i] = (t[i] + step * np.pi / 2) % (2 * np.pi)
                    else:
                        tr[i - n] = (tr[i - n] + step * np.pi / 2) % (2 * np.pi)
                    v = chi2(t, tr)
                    if v < cur - 1e-9:
                        sh, rs, cur, improved = t, tr, v, True
        if best is None or cur < best[0]:
            best = (cur, sh.copy(), rs.copy())
    return best


def refine_within_paper_errors(G, shifts, rshifts, prior_scale=1.0):
    """Continuous refinement of the paper's couplings inside their published errors.

    Free: |Lambda| and phase of the constrained waves (``LAMBDA_ERR``), the two
    Table 9 ratios' magnitude/phase; everything else stays at the paper's value.
    Gaussian penalties ((x - x_paper)/sigma) keep the solution within the total
    (stat + syst) uncertainties. Returns couplings, ratios, (chi2_tables, chi2_prior).
    """
    from scipy.optimize import least_squares

    base_c = np.array([PAPER_LAMBDA[n] for n in NAMES])
    free = [NAMES.index(n) for n in LAMBDA_ERR]
    mag0 = np.abs(base_c[free]); ph0 = np.angle(base_c[free]) + np.asarray(shifts)[free]
    sig_m = np.array([LAMBDA_ERR[NAMES[i]][0] for i in free]) * prior_scale
    sig_p = np.array([LAMBDA_ERR[NAMES[i]][1] for i in free]) * prior_scale
    r_paper = np.array([PAPER_RATIOS["a1"][1], PAPER_RATIOS["a1"][2]])
    r0m, r0p = np.abs(r_paper), np.angle(r_paper) + np.asarray(rshifts)[:2]
    rs_m = np.array([RATIO_ERR["a1_D"][0], RATIO_ERR["a1_f2"][0]]) * prior_scale
    rs_p = np.array([RATIO_ERR["a1_D"][1], RATIO_ERR["a1_f2"][1]]) * prior_scale
    nf = len(free)

    def unpack(x):
        c = base_c * bose_weights() * np.exp(1j * np.asarray(shifts))
        mags, phs = x[:nf], x[nf : 2 * nf]
        c[free] = mags * bose_weights()[free] * np.exp(1j * phs)
        rm, rp = x[2 * nf : 2 * nf + 2], x[2 * nf + 2 : 2 * nf + 4]
        a1 = (1.0 + 0j, rm[0] * np.exp(1j * rp[0]), rm[1] * np.exp(1j * rp[1]), 1.0 + 0j)
        pi = tuple(np.array(PAPER_RATIOS["pi1300"]) * np.exp(1j * np.r_[0.0, rshifts[3]]))
        return c, {"a1": a1, "pi1300": pi}

    def residual(x, with_prior=True):
        c, ratios = unpack(x)
        res = _table_residual(*fractions_from_matrix(c, raw_matrix(G, ratios)))
        if not with_prior:
            return res
        pri = np.concatenate([
            (x[:nf] - mag0) / sig_m, (x[nf : 2 * nf] - ph0) / sig_p,
            (x[2 * nf : 2 * nf + 2] - r0m) / rs_m, (x[2 * nf + 2 : 2 * nf + 4] - r0p) / rs_p,
        ])
        return np.concatenate([res, pri])

    x0 = np.concatenate([mag0, ph0, r0m, r0p])
    sol = least_squares(residual, x0)
    c, ratios = unpack(sol.x)
    tab = residual(sol.x, with_prior=False)
    return c, ratios, (float(tab @ tab), float(sol.cost * 2 - tab @ tab))


# ------------------------------------------------------------- toys and fits
def coefficient_objects(c, *, free=False, start=None, bound=4.0):
    """RealImag coefficients; with ``free`` every component except a1+ floats."""
    from jaxpwa import Parameter, RealImag

    start = c if start is None else start
    out = {}
    for i, name in enumerate(NAMES):
        if i == 0 or not free:
            out[name] = RealImag(float(c[i].real), float(c[i].imag))
        else:
            out[name] = RealImag(
                Parameter.coefficient(f"{name}.x", float(start[i].real), bounds=(-bound, bound), step=0.02),
                Parameter.coefficient(f"{name}.y", float(start[i].imag), bounds=(-bound, bound), step=0.02),
            )
    return out


def make_toy_sampler(model, c, *, batch=100_000, pool=500_000, margin=1.5, seed=0):
    """Exact accept-reject toy sampler for ``model`` with true complex coefficients ``c``.

    The envelope is ``margin`` times the maximum of ``weight * intensity`` over a
    large phase-space pool; every proposal batch is checked against it and
    sampling fails loudly (never clips) if the envelope is ever exceeded.
    Returns ``sample(size, seed) -> (NBodySample, n_proposals)`` and the envelope.
    """
    import jax

    generator = NBodyPhaseSpaceMC(M_D0, (M_PI,) * 4)
    template = model.prepare_cache(model.normalization_sample.take(jnp.array([0])))
    scales = template.component_scales
    functions = tuple(comp.function for comp in model.amplitude_model.components)
    coefficients = jnp.asarray(c)

    @jax.jit
    def target(p, w):
        amplitude = sum(
            scales[i] * coefficients[i] * f({"momenta": p}) for i, f in enumerate(functions)
        )
        return w * jnp.abs(amplitude) ** 2

    probe = generator.generate(pool, seed=seed + 10_000)
    envelope = float(margin * jnp.max(jnp.concatenate(
        [target(probe.momenta[i : i + batch], probe.weights[i : i + batch])
         for i in range(0, pool, batch)]
    )))

    def sample(size, seed=1):
        key = jax.random.PRNGKey(seed)
        accepted, count, generated = [], 0, 0
        while count < size:
            key, ps_key, u_key = jax.random.split(key, 3)
            batch_sample = generator.generate(batch, key=ps_key)
            t = target(batch_sample.momenta, batch_sample.weights)
            if not bool(jnp.all(jnp.isfinite(t) & (t <= envelope))):
                raise RuntimeError("toy envelope violated; increase margin")
            mask = jax.random.uniform(u_key, (batch,)) * envelope < t
            accepted.append(batch_sample.momenta[mask])
            count += int(mask.sum())
            generated += batch
        return NBodySample(jnp.concatenate(accepted)[:size], jnp.ones(size)), generated

    return sample, envelope


def component_matrix(model, sample, *, batch=100_000):
    """Normalized component amplitudes F_i = scale_i * f_i(p), shape (events, n)."""
    import jax

    template = model.prepare_cache(model.normalization_sample.take(jnp.array([0])))
    scales = template.component_scales
    functions = tuple(comp.function for comp in model.amplitude_model.components)

    @jax.jit
    def evaluate(p):
        return jnp.stack(
            [scales[i] * f({"momenta": p}) for i, f in enumerate(functions)], axis=-1
        )

    return np.concatenate(
        [np.asarray(evaluate(sample.momenta[i : i + batch]))
         for i in range(0, sample.size, batch)]
    )


def hermitian_matrix(F, w):
    """M_ij = mean(w * conj(F_i) F_j), the package's integral convention."""
    return (np.conj(F).T * np.asarray(w)) @ F / len(w)


def cp_even_fraction(c, X, M_union):
    """F+ = int|A+|^2 / int(|A+|^2 + |A-|^2), A+- = (A +- Abar)/sqrt2, Abar(p)=A(pbar).

    ``X_ij = mean(w * F_i(p) conj(F_j(pbar)))`` and ``M_union`` are evaluated on a
    CP-symmetric sample, so int|Abar|^2 = int|A|^2 and F+ = (1 + Re<A,Abar>/<A,A>)/2.
    """
    c = np.asarray(c, dtype=complex)
    cross = np.real(c @ X @ np.conj(c))
    norm = np.real(np.conj(c) @ M_union @ c)
    return 0.5 * (1.0 + cross / norm)


def mass_observables(momenta):
    """Paper Fig. 6-style variables; every pairing of an event is filled."""
    p = np.asarray(momenta)

    def m(*idx):
        q = p[:, list(idx), :].sum(axis=1)
        return np.sqrt(np.maximum(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(axis=1), 0))

    return {
        "m(pi+ pi-) [GeV]": np.concatenate([m(0, 1), m(0, 3), m(2, 1), m(2, 3)]),
        "m(pi+ pi+ pi-) [GeV]": np.concatenate([m(0, 2, 1), m(0, 2, 3)]),
        "m(pi- pi- pi+) [GeV]": np.concatenate([m(1, 3, 0), m(1, 3, 2)]),
        "m(pi+ pi+) [GeV]": m(0, 2),
    }, (4, 2, 2, 1)
