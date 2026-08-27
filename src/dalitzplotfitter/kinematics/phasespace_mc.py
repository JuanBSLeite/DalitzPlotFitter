"""Weighted three-body Monte Carlo generation with the ``phasespace`` package."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .sample import PhaseSpaceSample
from .vectors import invariant_mass_squared


def _to_energy_first(momentum) -> jnp.ndarray:
    """Convert ``phasespace`` ``(px, py, pz, E)`` vectors to ``(E, px, py, pz)``."""

    array = np.asarray(momentum)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(
            "phasespace momenta must have shape (N, 4) in (px, py, pz, E) order"
        )
    return jnp.asarray(array[:, [3, 0, 1, 2]])


@dataclass(frozen=True)
class PhasespaceMC:
    """Generate weighted three-body phase-space events using ``phasespace``.

    TensorFlow is used only inside ``phasespace.generate``. Returned tensors are
    immediately converted to JAX arrays, and no TensorFlow object enters the
    likelihood or amplitude evaluation.

    Raw, unnormalized phase-space weights are requested deliberately so samples
    can be used directly in Monte Carlo integration.
    """

    mother_mass: float
    masses: tuple[float, float, float]
    names: tuple[str, str, str] = ("p1", "p2", "p3")

    def __post_init__(self) -> None:
        if min(self.masses) < 0.0:
            raise ValueError("Final-state masses must be non-negative")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")
        if len(set(self.names)) != 3:
            raise ValueError("phasespace child names must be unique")

    def generate(self, size: int, *, seed: int | None = None) -> PhaseSpaceSample:
        """Return weighted phase-space events in the fitter's conventions."""

        if size <= 0:
            raise ValueError("size must be positive")

        import phasespace

        decay = phasespace.nbody_decay(
            self.mother_mass,
            list(self.masses),
            top_name="mother",
            names=list(self.names),
        )
        weights, _max_weights, particles = decay.generate(
            n_events=size,
            normalize_weights=False,
            seed=seed,
        )

        p1 = _to_energy_first(particles[self.names[0]])
        p2 = _to_energy_first(particles[self.names[1]])
        p3 = _to_energy_first(particles[self.names[2]])

        return PhaseSpaceSample(
            s12=invariant_mass_squared(p1 + p2),
            s13=invariant_mass_squared(p1 + p3),
            s23=invariant_mass_squared(p2 + p3),
            weights=jnp.asarray(np.asarray(weights)),
            p1=p1,
            p2=p2,
            p3=p3,
        )
