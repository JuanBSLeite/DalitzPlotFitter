"""QRules reaction construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReactionBuilder:
    """Small public wrapper around qrules.generate_transitions."""

    initial_state: Any
    final_state: list[Any]
    allowed_intermediate_particles: list[str] | None = None
    allowed_interaction_types: Any = None
    formalism: str = "canonical-helicity"
    mass_conservation_factor: float | None = 3.0
    max_angular_momentum: int = 2
    max_spin_magnitude: float = 2.0

    def build(self):
        import qrules

        return qrules.generate_transitions(
            initial_state=self.initial_state,
            final_state=self.final_state,
            allowed_intermediate_particles=self.allowed_intermediate_particles,
            allowed_interaction_types=self.allowed_interaction_types,
            formalism=self.formalism,
            mass_conservation_factor=self.mass_conservation_factor,
            max_angular_momentum=self.max_angular_momentum,
            max_spin_magnitude=self.max_spin_magnitude,
        )
