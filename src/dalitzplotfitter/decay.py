"""High-level three-body decay-model construction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from typing import Literal

import jax.numpy as jnp
from particle import Particle

from dalitzplotfitter.amplitude import (
    AmplitudeComponent,
    CoherentAmplitudeModel,
    ConstantAmplitude,
    PreparedAmplitudeCache,
)
from dalitzplotfitter.dynamics import (
    CovariantAngular,
    RelativisticBreitWigner,
    ResonanceAmplitude,
    ResonanceContext,
)
from dalitzplotfitter.dynamics.context import resolve_value
from dalitzplotfitter.fit import Parameter, ParameterKind
from dalitzplotfitter.integration import DalitzGaussLegendreGrid, GridIntegrator
from dalitzplotfitter.kinematics import PhaseSpaceMC, PhaseSpaceSample, SquareDalitzGrid
from dalitzplotfitter.pdf import SignalPDF


def _particle(name: str) -> Particle:
    errors: list[Exception] = []
    for resolver in (Particle.from_evtgen_name, Particle.from_name):
        try:
            return resolver(name)
        except Exception as exc:
            errors.append(exc)
    raise ValueError(
        f"Could not resolve particle {name!r} with the particle package"
    ) from errors[-1]


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


def _collect_parameters(value: object) -> tuple[Parameter, ...]:
    """Recursively collect fit Parameters from model declarations/plugins."""

    if isinstance(value, Parameter):
        return (value,)
    parameters = getattr(value, "parameters", None)
    if parameters is not None and not callable(parameters):
        if isinstance(parameters, dict):
            return tuple(
                item for item in parameters.values() if isinstance(item, Parameter)
            )
        try:
            return tuple(item for item in parameters if isinstance(item, Parameter))
        except TypeError:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        found = []
        for field in fields(value):
            found.extend(_collect_parameters(getattr(value, field.name)))
        return tuple(found)
    if isinstance(value, dict):
        return tuple(
            parameter
            for item in value.values()
            for parameter in _collect_parameters(item)
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            parameter for item in value for parameter in _collect_parameters(item)
        )
    return ()


def _validate_positive_quantity(value: object, label: str, *, allow_zero: bool) -> None:
    parameter = value if isinstance(value, Parameter) else None
    nominal = float(parameter.value if parameter is not None else value)
    invalid = nominal < 0.0 if allow_zero else nominal <= 0.0
    if invalid:
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {comparator}, got {nominal}")

    if parameter is not None and parameter.bounds is not None:
        lower, _ = parameter.bounds
        if lower is not None:
            invalid_lower = lower < 0.0 if allow_zero else lower <= 0.0
            if invalid_lower:
                comparator = "non-negative" if allow_zero else "positive"
                raise ValueError(
                    f"lower bound for {label} must be {comparator}, got {lower}"
                )


@dataclass(frozen=True)
class DecayChannel:
    """Parent particle and ordered three-body final state."""

    parent: str
    final_state: tuple[str, str, str]

    def __post_init__(self) -> None:
        if len(self.final_state) != 3:
            raise ValueError("DecayChannel requires exactly three final-state particles")
        if self.parent_mass <= sum(self.daughter_masses):
            raise ValueError(
                "parent mass must exceed the sum of final-state masses for a physical three-body decay"
            )

    @property
    def parent_mass(self) -> float:
        return _mass_gev(self.parent)

    @property
    def daughter_masses(self) -> tuple[float, float, float]:
        return tuple(_mass_gev(name) for name in self.final_state)

    @property
    def final_state_ids(self) -> tuple[int, int, int]:
        return tuple(int(_particle(name).pdgid) for name in self.final_state)


@dataclass(frozen=True)
class Resonance:
    """Declarative resonance component with interchangeable dynamics plugins."""

    name: str
    pair: tuple[int, int]
    coefficient: object
    lineshape: object = RelativisticBreitWigner()
    angular: object = CovariantAngular()
    mass: object | None = None
    width: object | None = None
    spin: int | None = None
    resonance_radius: object = 1.5
    parent_radius: object = 5.0

    def __post_init__(self) -> None:
        if len(set(self.pair)) != 2 or any(
            index not in (0, 1, 2) for index in self.pair
        ):
            raise ValueError(
                "resonance pair must contain two distinct indices from 0, 1, 2"
            )
        if self.mass is not None:
            _validate_positive_quantity(self.mass, f"{self.name}.mass", allow_zero=False)
        if self.width is not None:
            _validate_positive_quantity(self.width, f"{self.name}.width", allow_zero=True)
        _validate_positive_quantity(self.resonance_radius, f"{self.name}.resonance_radius", allow_zero=True)
        _validate_positive_quantity(self.parent_radius, f"{self.name}.parent_radius", allow_zero=True)
        if self.spin is not None and (self.spin < 0 or int(self.spin) != self.spin):
            raise ValueError("resonance spin must be a non-negative integer")


@dataclass(frozen=True)
class NonResonant:
    coefficient: object
    name: str = "NR"


@dataclass(frozen=True)
class DalitzAmplitude:
    """Direct amplitude depending on two Dalitz coordinates.

    This declaration is intended for dynamics such as ``QMI2D`` that are not a
    one-dimensional isobar lineshape and therefore act directly on the event's
    Dalitz invariants.
    """

    name: str
    dynamics: object
    coefficient: object


@dataclass(frozen=True)
class _ResolvedDirectDynamics:
    dynamics: object

    def __call__(self, data, parameters=None):
        return resolve_value(self.dynamics, parameters)(data)


@dataclass(frozen=True, init=False)
class DecayModel:
    """Build a coherent model with deterministic quadrature normalization."""

    channel: DecayChannel
    components: tuple[Resonance | NonResonant | DalitzAmplitude, ...]
    normalize_components: bool
    normalization_resolution: int
    normalization_method: Literal["gauss-legendre", "square-dalitz"]
    normalization_pair: tuple[int, int]
    normalization_bin_width: float
    normalization_order_m13: int | None
    normalization_order_m23: int | None
    _normalization_sample: PhaseSpaceSample | None

    def __init__(
        self,
        channel: DecayChannel,
        components: Iterable[Resonance | NonResonant | DalitzAmplitude],
        *,
        normalize_components: bool = True,
        normalization_resolution: int = 1000,
        normalization_method: Literal["gauss-legendre", "square-dalitz"] = "gauss-legendre",
        normalization_pair: tuple[int, int] = (0, 1),
        normalization_bin_width: float = 0.005,
        normalization_order_m13: int | None = None,
        normalization_order_m23: int | None = None,
    ) -> None:
        if normalization_resolution < 2:
            raise ValueError("normalization_resolution must be at least 2")
        if normalization_method not in ("gauss-legendre", "square-dalitz"):
            raise ValueError(
                "normalization_method must be either 'gauss-legendre' or 'square-dalitz'"
            )
        if len(set(normalization_pair)) != 2 or any(
            index not in (0, 1, 2) for index in normalization_pair
        ):
            raise ValueError(
                "normalization_pair must contain two distinct indices from 0, 1, 2"
            )
        if normalization_bin_width <= 0.0:
            raise ValueError("normalization_bin_width must be positive")
        if normalization_order_m13 is not None and normalization_order_m13 < 2:
            raise ValueError("normalization_order_m13 must be at least 2")
        if normalization_order_m23 is not None and normalization_order_m23 < 2:
            raise ValueError("normalization_order_m23 must be at least 2")
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "normalize_components", bool(normalize_components))
        object.__setattr__(self, "normalization_resolution", int(normalization_resolution))
        object.__setattr__(self, "normalization_method", normalization_method)
        object.__setattr__(self, "normalization_pair", tuple(normalization_pair))
        object.__setattr__(self, "normalization_bin_width", float(normalization_bin_width))
        object.__setattr__(self, "normalization_order_m13", normalization_order_m13)
        object.__setattr__(self, "normalization_order_m23", normalization_order_m23)
        object.__setattr__(self, "_normalization_sample", None)
        if not self.components:
            raise ValueError("DecayModel requires at least one amplitude component")
        names = [component.name for component in self.components]
        if len(set(names)) != len(names):
            raise ValueError("DecayModel component names must be unique")
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        names: dict[str, Parameter] = {}
        dynamics_slots: dict[tuple[str, str], Parameter] = {}
        for component in self.components:
            for parameter in _collect_parameters(component):
                if parameter.name in names and names[parameter.name] != parameter:
                    raise ValueError(f"Conflicting definitions for parameter {parameter.name!r}")
                names[parameter.name] = parameter
                if parameter.kind is ParameterKind.DYNAMICS:
                    if parameter.owner != component.name:
                        raise ValueError(
                            f"Dynamics parameter {parameter.name!r} must have owner={component.name!r}"
                        )
                    backend_key = parameter.backend_name or parameter.name
                    slot = (component.name, backend_key)
                    previous = dynamics_slots.get(slot)
                    if previous is not None and previous.name != parameter.name:
                        raise ValueError(
                            "Dynamics parameters "
                            f"{previous.name!r} and {parameter.name!r} map to the same "
                            f"backend key {backend_key!r} for component {component.name!r}"
                        )
                    dynamics_slots[slot] = parameter
                if parameter.kind is ParameterKind.COEFFICIENT and parameter.owner is not None and parameter.owner != component.name:
                    raise ValueError(
                        f"Coefficient parameter {parameter.name!r} must have owner={component.name!r}"
                    )

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        unique: dict[str, Parameter] = {}
        for component in self.components:
            for parameter in _collect_parameters(component):
                unique.setdefault(parameter.name, parameter)
        return tuple(unique.values())

    @property
    def normalization_sample(self) -> PhaseSpaceSample:
        sample = self._normalization_sample
        if sample is None:
            if self.normalization_method == "gauss-legendre":
                sample = DalitzGaussLegendreGrid(
                    self.channel.parent_mass,
                    self.channel.daughter_masses,
                    bin_width=self.normalization_bin_width,
                    order_m13=self.normalization_order_m13,
                    order_m23=self.normalization_order_m23,
                ).sample()
            else:
                sample = SquareDalitzGrid(
                    self.channel.parent_mass,
                    self.channel.daughter_masses,
                    resolution=self.normalization_resolution,
                    pair=self.normalization_pair,
                    quadrature="gauss-legendre",
                ).sample()
            object.__setattr__(self, "_normalization_sample", sample)
        return sample

    def _build_resonance(self, component: Resonance) -> AmplitudeComponent:
        i, j = component.pair
        bachelor = next(index for index in range(3) if index not in component.pair)
        masses = self.channel.daughter_masses
        mass0 = component.mass if component.mass is not None else _mass_gev(component.name)
        width0 = component.width if component.width is not None else _width_gev(component.name)
        spin = component.spin if component.spin is not None else _spin(component.name)
        context = ResonanceContext(
            parent_mass=self.channel.parent_mass,
            daughter_masses=(masses[i], masses[j]),
            bachelor_mass=masses[bachelor],
            spin=spin,
            pole_mass=mass0,
            pole_width=width0,
            resonance_radius=component.resonance_radius,
            parent_radius=component.parent_radius,
        )
        dynamics = ResonanceAmplitude(
            context=context,
            daughter_key=f"p{i + 1}",
            partner_key=f"p{j + 1}",
            bachelor_key=f"p{bachelor + 1}",
            final_state=self.channel.final_state_ids,
            lineshape=component.lineshape,
            angular=component.angular,
        )
        return AmplitudeComponent(component.name, dynamics, component.coefficient)

    @property
    def amplitude_model(self) -> CoherentAmplitudeModel:
        built = []
        for component in self.components:
            if isinstance(component, Resonance):
                built.append(self._build_resonance(component))
            elif isinstance(component, NonResonant):
                built.append(AmplitudeComponent(component.name, ConstantAmplitude(), component.coefficient))
            elif isinstance(component, DalitzAmplitude):
                built.append(
                    AmplitudeComponent(
                        component.name,
                        _ResolvedDirectDynamics(component.dynamics),
                        component.coefficient,
                    )
                )
            else:
                raise TypeError(f"Unsupported amplitude component: {type(component)!r}")
        return CoherentAmplitudeModel(tuple(built))

    def generate_phase_space(self, size: int, *, seed: int | None = None) -> PhaseSpaceSample:
        return PhaseSpaceMC(self.channel.parent_mass, self.channel.daughter_masses).generate(size, seed=seed)

    def _component_scale(self, component: AmplitudeComponent, values=None):
        if not self.normalize_components:
            return 1.0
        sample = self.normalization_sample
        raw = jnp.asarray(component.function(sample.as_dict(), values))
        integral = jnp.mean(sample.weights * jnp.abs(raw) ** 2)
        return 1.0 / jnp.sqrt(integral)

    def amplitude(self, data, values=None):
        total = None
        for component in self.amplitude_model.components:
            dynamics = jnp.asarray(component.function(data, values))
            coefficient = jnp.asarray(component.coefficient.value(values))
            component_values = coefficient * self._component_scale(component, values) * dynamics
            total = component_values if total is None else total + component_values
        return jnp.asarray(total)

    def intensity(self, data, values=None):
        amplitude = self.amplitude(data, values)
        return jnp.real(amplitude * jnp.conj(amplitude))

    def pdf(self, normalization_sample: PhaseSpaceSample | None = None, *, efficiency=None) -> SignalPDF:
        sample = self.normalization_sample if normalization_sample is None else normalization_sample
        def intensity(data, parameters):
            return self.intensity(data, parameters)
        kwargs = {}
        if efficiency is not None:
            kwargs["efficiency"] = efficiency
        return SignalPDF(intensity=intensity, integrator=GridIntegrator(sample), **kwargs)

    def prepare_cache(
        self,
        data_sample: PhaseSpaceSample,
        normalization_sample: PhaseSpaceSample | None = None,
        *,
        efficiency_normalization=None,
        normalize_components: bool | None = None,
    ) -> PreparedAmplitudeCache:
        sample = self.normalization_sample if normalization_sample is None else normalization_sample
        normalize = self.normalize_components if normalize_components is None else bool(normalize_components)
        return PreparedAmplitudeCache.prepare(
            self.amplitude_model.components,
            data=data_sample.as_dict(),
            normalization_data=sample.as_dict(),
            normalization_weights=sample.weights,
            parameters=self.parameters,
            efficiency_normalization=efficiency_normalization,
            normalize_components=normalize,
        )

    def _fraction_cache(
        self,
        normalization_sample: PhaseSpaceSample | None,
        efficiency,
    ) -> PreparedAmplitudeCache:
        sample = (
            self.normalization_sample
            if normalization_sample is None
            else normalization_sample
        )
        efficiency_values = None
        if efficiency is not None:
            efficiency_values = (
                efficiency(sample.as_dict()) if callable(efficiency) else efficiency
            )
            efficiency_values = jnp.asarray(efficiency_values)
            if efficiency_values.shape != (sample.size,):
                raise ValueError(
                    "efficiency must return one value per normalization point"
                )
        return self.prepare_cache(
            sample,
            normalization_sample=sample,
            efficiency_normalization=efficiency_values,
        )

    def fit_fractions(
        self,
        fit_values=None,
        *,
        normalization_sample: PhaseSpaceSample | None = None,
        efficiency=None,
    ):
        """Return component fit fractions at a parameter point.

        Fractions are physical (efficiency excluded) by default. Supplying an
        efficiency returns acceptance-weighted fractions instead.
        """

        values = {} if fit_values is None else fit_values
        return self._fraction_cache(
            normalization_sample, efficiency
        ).fit_fractions(values)

    def interference_fractions(
        self,
        fit_values=None,
        *,
        normalization_sample: PhaseSpaceSample | None = None,
        efficiency=None,
    ):
        """Return pairwise interference fractions at a parameter point."""

        values = {} if fit_values is None else fit_values
        return self._fraction_cache(
            normalization_sample, efficiency
        ).interference_fractions(values)

    def print_fit_fractions(
        self,
        fit_values=None,
        *,
        normalization_sample: PhaseSpaceSample | None = None,
        efficiency=None,
        include_interference: bool = False,
        precision: int = 3,
    ) -> dict[str, float]:
        """Print fit fractions as percentages and return them by component name."""

        if precision < 0:
            raise ValueError("precision must be non-negative")
        values = {} if fit_values is None else fit_values
        cache = self._fraction_cache(normalization_sample, efficiency)
        fractions = cache.fit_fractions(values)
        result = {
            component.name: float(fractions[index])
            for index, component in enumerate(cache.components)
        }
        convention = "acceptance-weighted" if efficiency is not None else "physical"
        print(f"Fit fractions ({convention})")
        print(f"{'component':24s} {'fraction [%]':>16s}")
        for name, fraction in result.items():
            print(f"{name:24s} {100.0 * fraction:16.{precision}f}")
        print(f"{'sum':24s} {100.0 * sum(result.values()):16.{precision}f}")

        if include_interference:
            interference = cache.interference_fractions(values)
            print("\nInterference fractions")
            print(f"{'pair':49s} {'fraction [%]':>16s}")
            for i, first in enumerate(cache.components):
                for j in range(i + 1, len(cache.components)):
                    second = cache.components[j]
                    fraction = float(interference[i, j])
                    print(
                        f"{first.name + ' x ' + second.name:49s} "
                        f"{100.0 * fraction:16.{precision}f}"
                    )
        return result
