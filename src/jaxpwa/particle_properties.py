"""Particle-database lookups shared by decay-channel declarations."""

from __future__ import annotations

from particle import Particle


def resolve_particle(name: str) -> Particle:
    """Resolve an EvtGen or PDG particle name with a stable public error."""
    if not isinstance(name, str) or not name:
        raise ValueError("particle name must be a nonempty string")
    errors: list[Exception] = []
    for resolver in (Particle.from_evtgen_name, Particle.from_name):
        try:
            return resolver(name)
        except Exception as exc:
            errors.append(exc)
    raise ValueError(
        f"Could not resolve particle {name!r} with the particle package"
    ) from errors[-1]


def mass_gev(name: str) -> float:
    """Return the database mass in GeV."""
    particle = resolve_particle(name)
    if particle.mass is None:
        raise ValueError(f"Particle {name!r} has no mass in the particle database")
    return float(particle.mass) / 1000.0


def width_gev(name: str) -> float:
    """Return the database width in GeV."""
    particle = resolve_particle(name)
    if particle.width is None:
        raise ValueError(f"Particle {name!r} has no width in the particle database")
    return float(particle.width) / 1000.0


def integer_spin(name: str, *, context: str = "resonance") -> int:
    """Return integer J, rejecting missing and half-integer database values."""
    particle = resolve_particle(name)
    if particle.J is None:
        raise ValueError(f"Particle {name!r} has no spin in the particle database")
    spin = float(particle.J)
    rounded = round(spin)
    if abs(spin - rounded) > 1e-12:
        raise ValueError(f"{context} requires integer spin, got J={spin} for {name!r}")
    return int(rounded)


def resolve_resonance_properties(
    name: str,
    *,
    mass: object | None = None,
    width: object | None = None,
    spin: int | None = None,
    context: str = "resonance",
    validate_name: bool = True,
) -> tuple[object, object, int]:
    """Resolve nominal resonance properties with optional explicit overrides.

    ``validate_name=False`` preserves the legacy three-body convention where a
    fully specified resonance may use an arbitrary component label. Whenever a
    property is missing, the name is still resolved from ``particle``.
    """
    particle = None
    if validate_name or mass is None or width is None or spin is None:
        particle = resolve_particle(name)

    if mass is None:
        if particle.mass is None:
            raise ValueError(f"Particle {name!r} has no mass in the particle database")
        mass = float(particle.mass) / 1000.0
    if width is None:
        if particle.width is None:
            raise ValueError(f"Particle {name!r} has no width in the particle database")
        width = float(particle.width) / 1000.0
    if spin is None:
        if particle.J is None:
            raise ValueError(f"Particle {name!r} has no spin in the particle database")
        value = float(particle.J)
        rounded = round(value)
        if abs(value - rounded) > 1e-12:
            raise ValueError(
                f"{context} requires integer spin, got J={value} for {name!r}"
            )
        spin = int(rounded)
    elif isinstance(spin, bool) or not isinstance(spin, int) or spin < 0:
        raise ValueError(f"{context} spin must be a non-negative integer")
    return mass, width, spin


__all__ = [
    "integer_spin",
    "mass_gev",
    "resolve_particle",
    "resolve_resonance_properties",
    "width_gev",
]
