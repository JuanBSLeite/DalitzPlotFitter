"""High-level three-body decay-model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from particle import Particle

from dalitzplotfitter.amplitude import (
    AmplitudeComponent,
    CoherentAmplitudeModel,
    ConstantAmplitude,
)
from dalitzplotfitter.dynamics import ResonanceAmplitude
from dalitzplotfitter.integration import MonteCarloIntegrator
from dalitzplotfitter.kinematics import PhasespaceMC, PhaseSpaceSample
from dalitzplotfitter.pdf import SignalPDF


def _particle(name: str) -> Particle:
    """Resolve a particle using common Particle package naming conventions."""

    errors: list[Exception] = []
    for resolver in (Particle.from_evtgen_name, Particle.from_name):
        try:
            return resolver(name)
        except Exception as exc:  # Particle raises different lookup exceptions.
            errors.append(exc)
    raise ValueError(f"Could not resolve particle {name!r} with the particle package") from errors[-1]


def _mass_gev(name: str) -> float:
    particle = _particle(name)
    if particle.mass is None:
        raise ValueError(f"Particle {name!r} has no mass in the particle database")
    return float(particle.mass) / 1000.0


def _width_gev(name: str) -> float:
    particle = _particle(name)
    if particle.width is None:
        raise ValueError(f"Particle {name!r} has no width in the particle database")
    return float(particle.width) / 1000.0


def _spin(name: str) -> int:
    particle = _particle(name)
    if particle.J is None:
        raise ValueError(f"Particle {name!r} has no spin in the particle database")
    spin = float(particle.J)
    rounded = round(spin)
    if abs(spin - rounded) > 1e-12:
        raise ValueError(
            f"Three-spinless-body resonance component requires integer spin, got J={spin} for {name!r}"
        )
    return int(rounded)


@dataclass(frozen=True)
class DecayChannel:
    """Parent particle and ordered three-body final state."""

    parent: str
    final_state: tuple[str, str, str]

    def __post_init__(self) -> None:
        if len(self.final_state) != 3:
            raise ValueError("DecayChannel requires exactly three final-state particles")

    @property
    def parent_mass(self) -> float:
        return _mass_gev(self.parent)

    @property
    def daughter_masses(self) -> tuple[float, float, float]:
        return tuple(_mass_gev(name) for name in self.final_state)


@dataclass(frozen=True)
class Resonance:
    """Declarative resonance component.

    ``pair`` contains zero-based indices into ``DecayChannel.final_state``.
    Mass, width and spin default to values from the ``particle`` package and can
    be overridden for a specific analysis or historical model.
    """

    name: str
    pair: tuple[int, int]
    coefficient: object
    mass: float | None = None
    width: float | None = None
    spin: int | None = None
    resonance_radius: float = 1.5
    parent_radius: float = 5.0

    def __post_init__(self) -> None:
        if len(set(self.pair)) != 2 or any(index not in (0, 1, 2) for index in self.pair):
            raise ValueError("resonance pair must contain two distinct indices from 0, 1, 2")


@dataclass(frozen=True)
class NonResonant:
    """Constant non-resonant component."""

    coefficient: object
    name: str = "NR"


@dataclass(frozen=True)
class DecayModel:
    """Build a coherent amplitude and normalized PDF from a decay channel."""

    channel: DecayChannel
    components: tuple[Resonance | NonResonant, ...]

    def __init__(
        self,
        channel: DecayChannel,
        components: Iterable[Resonance | NonResonant],
    ) -> None:
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "components", tuple(components))
        if not self.components:
            raise ValueError("DecayModel requires at least one amplitude component")

    def _build_resonance(self, component: Resonance) -> AmplitudeComponent:
        i, j = component.pair
        bachelor = next(index for index in range(3) if index not in component.pair)
        masses = self.channel.daughter_masses

        mass0 = component.mass if component.mass is not None else _mass_gev(component.name)
        width0 = component.width if component.width is not None else _width_gev(component.name)
        spin = component.spin if component.spin is not None else _spin(component.name)

        dynamics = ResonanceAmplitude(
            mass0=mass0,
            width0=width0,
            parent_mass=self.channel.parent_mass,
            daughter_masses=(masses[i], masses[j]),
            bachelor_mass=masses[bachelor],
            angular_momentum=spin,
            resonance_radius=component.resonance_radius,
            parent_radius=component.parent_radius,
            daughter_key=f"p{i + 1}",
            partner_key=f"p{j + 1}",
            bachelor_key=f"p{bachelor + 1}",
            final_state=self.channel.final_state,
        )
        return AmplitudeComponent(component.name, dynamics, component.coefficient)

    @property
    def amplitude_model(self) -> CoherentAmplitudeModel:
        built = []
        for component in self.components:
            if isinstance(component, Resonance):
                built.append(self._build_resonance(component))
            elif isinstance(component, NonResonant):
                built.append(
                    AmplitudeComponent(
                        component.name,
                        ConstantAmplitude(),
                        component.coefficient,
                    )
                )
            else:
                raise TypeError(f"Unsupported amplitude component: {type(component)!r}")
        return CoherentAmplitudeModel(tuple(built))

    def generate_phase_space(self, size: int, *, seed: int | None = None) -> PhaseSpaceSample:
        return PhasespaceMC(
            self.channel.parent_mass,
            self.channel.daughter_masses,
        ).generate(size, seed=seed)

    def amplitude(self, data, coefficient_values=None):
        return self.amplitude_model.amplitude(
            data,
            coefficient_values=coefficient_values,
        )

    def intensity(self, data, coefficient_values=None):
        return self.amplitude_model.intensity(
            data,
            coefficient_values=coefficient_values,
        )

    def pdf(self, normalization_sample: PhaseSpaceSample, *, efficiency=None) -> SignalPDF:
        """Build a normalized signal PDF on a fixed weighted MC sample."""

        def intensity(data, parameters):
            return self.intensity(data, coefficient_values=parameters)

        kwargs = {}
        if efficiency is not None:
            kwargs["efficiency"] = efficiency
        return SignalPDF(
            intensity=intensity,
            integrator=MonteCarloIntegrator(normalization_sample),
            **kwargs,
        )
