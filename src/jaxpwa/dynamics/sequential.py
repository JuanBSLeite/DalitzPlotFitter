"""LS-coupled sequential four-body amplitudes for spinless external particles.

Integer-spin Jacob-Wick helicity sums use d^j_(m,n)(theta) and exp(+i*m*phi)
for D*. Conventions and the LS-to-helicity coefficients are in docs/four_body.md.
No parent polarization or external-particle spin transport is implied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial, sqrt

import jax.numpy as jnp

from jaxpwa.kinematics.four_body import cascade_coordinates, pair_coordinates
from jaxpwa.kinematics.vectors import invariant_mass_squared
from jaxpwa.particle_properties import resolve_resonance_properties

from .context import ResonanceContext, resolve_value
from .lineshape.common import blatt_weisskopf_from_momenta, breakup_momentum
from .lineshape.pole import Pole
from .lineshape.relativistic_breit_wigner import RelativisticBreitWigner


def _angular_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
        raise ValueError(f"{label} must be an integer in 0..4")


def wigner_d(j, m, n, cos_theta):
    """Small Wigner d matrix for integer j, row m, column n (Condon-Shortley)."""
    if not all(isinstance(x, int) for x in (j, m, n)) or j < 0:
        raise ValueError("j, m, n must be integers with j >= 0")
    if abs(m) > j or abs(n) > j:
        return jnp.zeros_like(cos_theta)
    c = jnp.sqrt(jnp.maximum((1 + cos_theta) / 2, 0))
    s = jnp.sqrt(jnp.maximum((1 - cos_theta) / 2, 0))
    prefactor = sqrt(
        factorial(j + n) * factorial(j - n) * factorial(j + m) * factorial(j - m)
    )
    result = jnp.zeros_like(c)
    for k in range(max(0, n - m), min(j + n, j - m) + 1):
        den = (
            factorial(j + n - k)
            * factorial(k)
            * factorial(m - n + k)
            * factorial(j - m - k)
        )
        result = result + (
            (-1) ** (m - n + k)
            * prefactor
            / den
            * c ** (2 * j + n - m - 2 * k)
            * s ** (m - n + 2 * k)
        )
    return result


def clebsch_gordan(j1, m1, j2, m2, j, m):
    """Integer-spin Clebsch-Gordan coefficient, evaluated once on the host."""
    if m1 + m2 != m or abs(m) > j or not abs(j1 - j2) <= j <= j1 + j2:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2:
        return 0.0
    prefactor = sqrt(
        (2 * j + 1)
        * factorial(j + j1 - j2)
        * factorial(j - j1 + j2)
        * factorial(j1 + j2 - j)
        / factorial(j1 + j2 + j + 1)
        * factorial(j + m)
        * factorial(j - m)
        * factorial(j1 - m1)
        * factorial(j1 + m1)
        * factorial(j2 - m2)
        * factorial(j2 + m2)
    )
    total = 0.0
    for k in range(
        max(0, j2 - j - m1, j1 + m2 - j), min(j1 + j2 - j, j1 - m1, j2 + m2) + 1
    ):
        total += (-1) ** k / (
            factorial(k)
            * factorial(j1 + j2 - j - k)
            * factorial(j1 - m1 - k)
            * factorial(j2 + m2 - k)
            * factorial(j - j2 + m1 + k)
            * factorial(j - j1 - m2 + k)
        )
    return prefactor * total


def _radial(mass, m1, m2, orbital, radius):
    q = breakup_momentum(mass, m1, m2)
    # Raw barriers keep the definition valid over broad, off-shell isobars.
    return q**orbital * blatt_weisskopf_from_momenta(
        q, q, orbital, radius, normalize_at_pole=False
    )


@dataclass(frozen=True)
class Isobar:
    """Reusable resonance pole and lineshape; spin refers to the particle.

    ``orbital`` in evaluate() refers instead to the decay vertex. This matters
    for R -> S b: its running-width power is 2L+1, not necessarily 2J_R+1.
    Default: the existing fixed-width Pole. RelativisticBreitWigner and other
    existing lineshapes can be supplied for decays to stable daughters. A broad
    daughter isobar needs a dedicated width model; a two-stable-body RBW is
    deliberately rejected for the outer resonance of CascadeChain.
    """

    mass: object
    width: object
    spin: int = 0
    lineshape: object = field(default_factory=Pole)
    radius: object = 1.5
    particle_name: str | None = None

    def __post_init__(self):
        _angular_integer(self.spin, "spin")
        for name in ("mass", "width", "radius"):
            value = float(resolve_value(getattr(self, name)))
            if not jnp.isfinite(value) or value < 0 or (name == "mass" and value == 0):
                raise ValueError(f"{name} must be finite and physically nonnegative")
        if self.particle_name is not None and (
            not isinstance(self.particle_name, str) or not self.particle_name
        ):
            raise ValueError("particle_name must be a nonempty string or None")

    @classmethod
    def from_particle(
        cls,
        name: str,
        *,
        mass: object | None = None,
        width: object | None = None,
        spin: int | None = None,
        lineshape: object | None = None,
        radius: object = 1.5,
    ) -> Isobar:
        """Build an isobar from an EvtGen or PDG name in ``particle``.

        Database masses and widths are converted from MeV to GeV. Explicit
        overrides take precedence and may include fit ``Parameter`` objects for
        mass, width, and radius. Spin remains a static integer because it fixes
        the compiled angular sum. The default lineshape remains ``Pole``.
        """
        resolved_mass, resolved_width, resolved_spin = resolve_resonance_properties(
            name,
            mass=mass,
            width=width,
            spin=spin,
            context="Four-body isobar",
            validate_name=True,
        )
        return cls(
            mass=resolved_mass,
            width=resolved_width,
            spin=resolved_spin,
            lineshape=Pole() if lineshape is None else lineshape,
            radius=radius,
            particle_name=name,
        )

    def evaluate(self, mass, m1, m2, parent_mass, bachelor_mass, orbital, values):
        context = ResonanceContext(
            parent_mass=parent_mass,
            daughter_masses=(m1, m2),
            bachelor_mass=bachelor_mass,
            spin=orbital,
            pole_mass=resolve_value(self.mass, values),
            pole_width=resolve_value(self.width, values),
            resonance_radius=resolve_value(self.radius, values),
        )
        lineshape = resolve_value(self.lineshape, values)
        return lineshape(mass, context) * _radial(
            mass, m1, m2, orbital, context.resonance_radius
        )


def _masses(p):
    return jnp.sqrt(jnp.maximum(invariant_mass_squared(p), 0))


class _PreparedChain:
    """Freeze spin geometry before the mass/width/radius fit hot path."""

    @property
    def _prefix(self):
        return f"_four_body_{id(self)}_"

    def prepare_data(self, data):
        if self._prefix + "angular" in data:
            return data
        geometry = self._geometry(data["momenta"])
        return {**data, **{self._prefix + k: v for k, v in geometry.items()}}

    def compact_prepared_data(self, data):
        prepared = self.prepare_data(data)
        return {k: v for k, v in prepared.items() if k.startswith(self._prefix)}

    def _get_geometry(self, data):
        prepared = self.compact_prepared_data(data)
        return {k[len(self._prefix) :]: v for k, v in prepared.items()}


@dataclass(frozen=True)
class PairChain(_PreparedChain):
    """P -> R1 R2, R1 -> a b, R2 -> c d, with production L=S.

    Each instance is one LS wave. Combine different L waves coherently using
    separate AmplitudeComponents. Daughter order fixes all angular signs.
    """

    first: Isobar
    second: Isobar
    orbital: int = 0
    order: tuple[int, int, int, int] = (0, 1, 2, 3)
    parent_radius: object = 5.0

    def __post_init__(self):
        _angular_integer(self.orbital, "orbital")
        if (
            not abs(self.first.spin - self.second.spin)
            <= self.orbital
            <= (self.first.spin + self.second.spin)
        ):
            raise ValueError("production L=S must couple J1,J2 to a scalar parent")
        if tuple(sorted(self.order)) != (0, 1, 2, 3):
            raise ValueError("order must be a permutation of (0, 1, 2, 3)")

    def _geometry(self, momenta):
        p = momenta[:, self.order, :]
        angles = pair_coordinates(p)
        m1, m2 = jnp.sqrt(angles["s_ab"]), jnp.sqrt(angles["s_cd"])
        masses = _masses(p)
        mother = _masses(jnp.sum(p, axis=1))
        j1, j2, orbital = self.first.spin, self.second.spin, self.orbital
        angular = jnp.zeros_like(m1, dtype=complex)
        for h in range(-min(j1, j2), min(j1, j2) + 1):
            # R2's helicity axes use R_y(pi), so |j2,h> carries the
            # (-1)^(j2-h) phase relative to the common-axis |j2,-h> LS state.
            coupling = (-1) ** (orbital + j2 - h) * clebsch_gordan(
                j1, h, j2, -h, orbital, 0
            )
            angular += (
                coupling
                * wigner_d(j1, h, 0, angles["cos_theta1"])
                * wigner_d(j2, h, 0, angles["cos_theta2"])
                * jnp.exp(1j * h * angles["phi"])
            )
        return dict(m1=m1, m2=m2, masses=masses, mother=mother, angular=angular)

    def __call__(self, data, parameters=None):
        g = self._get_geometry(data)
        m1, m2, masses, mother = g["m1"], g["m2"], g["masses"], g["mother"]
        first = self.first.evaluate(
            m1, masses[:, 0], masses[:, 1], mother, m2, self.first.spin, parameters
        )
        second = self.second.evaluate(
            m2, masses[:, 2], masses[:, 3], mother, m1, self.second.spin, parameters
        )
        return (
            g["angular"]
            * first
            * second
            * _radial(
                mother,
                m1,
                m2,
                self.orbital,
                resolve_value(self.parent_radius, parameters),
            )
        )


@dataclass(frozen=True)
class CascadeChain(_PreparedChain):
    """P -> R a, R -> S b, S -> c d, with orbital L at R -> S b.

    Production orbital momentum is J_R; S -> c d has orbital momentum J_S.
    The R decay couples L and J_S to J_R. One instance is one LS wave.
    """

    outer: Isobar
    inner: Isobar
    orbital: int = 0
    order: tuple[int, int, int, int] = (0, 1, 2, 3)
    parent_radius: object = 5.0

    def __post_init__(self):
        if isinstance(self.outer.lineshape, RelativisticBreitWigner):
            raise ValueError(
                "R -> S b with unstable S needs Pole or a dedicated width model; "
                "the two-stable-body RelativisticBreitWigner is not supported"
            )
        _angular_integer(self.orbital, "orbital")
        if (
            not abs(self.orbital - self.inner.spin)
            <= self.outer.spin
            <= (self.orbital + self.inner.spin)
        ):
            raise ValueError("R -> S b requires L and J_S to couple to J_R")
        if tuple(sorted(self.order)) != (0, 1, 2, 3):
            raise ValueError("order must be a permutation of (0, 1, 2, 3)")

    def _geometry(self, momenta):
        p = momenta[:, self.order, :]
        angles = cascade_coordinates(p)
        mr, ms = jnp.sqrt(angles["s_bcd"]), jnp.sqrt(angles["s_cd"])
        masses = _masses(p)
        mother = _masses(jnp.sum(p, axis=1))
        jr, js, orbital = self.outer.spin, self.inner.spin, self.orbital
        angular = jnp.zeros_like(mr, dtype=complex)
        for h in range(-min(jr, js), min(jr, js) + 1):
            coupling = sqrt((2 * orbital + 1) / (2 * jr + 1)) * clebsch_gordan(
                orbital, 0, js, h, jr, h
            )
            angular += (
                coupling
                * wigner_d(jr, 0, h, angles["cos_theta_R"])
                * wigner_d(js, h, 0, angles["cos_theta_S"])
                * jnp.exp(1j * h * angles["phi"])
            )
        return dict(mr=mr, ms=ms, masses=masses, mother=mother, angular=angular)

    def __call__(self, data, parameters=None):
        g = self._get_geometry(data)
        mr, ms, masses, mother = g["mr"], g["ms"], g["masses"], g["mother"]
        outer = self.outer.evaluate(
            mr, ms, masses[:, 1], mother, masses[:, 0], self.orbital, parameters
        )
        inner = self.inner.evaluate(
            ms,
            masses[:, 2],
            masses[:, 3],
            mr,
            masses[:, 1],
            self.inner.spin,
            parameters,
        )
        return (
            g["angular"]
            * outer
            * inner
            * _radial(
                mother,
                mr,
                masses[:, 0],
                self.outer.spin,
                resolve_value(self.parent_radius, parameters),
            )
        )
