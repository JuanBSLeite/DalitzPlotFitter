"""Adapters that insert DalitzPlotFitter symbolic dynamics into AmpForm."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .laura import blatt_weisskopf_factor, relativistic_breit_wigner


@dataclass(frozen=True)
class LauraRelativisticBreitWignerBuilder:
    """Build a Laura++-convention relativistic Breit-Wigner for AmpForm.

    AmpForm continues to provide the decay topology, helicity formalism, angular
    terms and symmetrization. Only the resonance dynamics inserted through its
    ``DynamicsSelector`` are formulated by DalitzPlotFitter.

    The returned resonance term contains the Laura++ relativistic Breit-Wigner
    with mass-dependent width. When ``form_factor=True`` it is also multiplied
    by the resonance-decay Blatt-Weisskopf factor, while the same factor squared
    appears inside the running width.
    """

    form_factor: bool = True
    energy_dependent_width: bool = True
    meson_radius_default: float = 1.0

    def __call__(self, resonance, variable_pool):
        mass0 = sp.Symbol(f"m_{resonance.name}", nonnegative=True)
        gamma0 = sp.Symbol(f"Gamma_{resonance.name}", nonnegative=True)
        radius = sp.Symbol(f"d_{resonance.name}", nonnegative=True)

        mass = variable_pool.incoming_state_mass
        daughter_mass1 = variable_pool.outgoing_state_mass1
        daughter_mass2 = variable_pool.outgoing_state_mass2
        angular_momentum = variable_pool.angular_momentum
        if angular_momentum is None:
            angular_momentum = 0

        width_radius = radius if self.form_factor else None
        expression = relativistic_breit_wigner(
            mass=mass,
            mass0=mass0,
            gamma0=gamma0,
            daughter_mass1=daughter_mass1,
            daughter_mass2=daughter_mass2,
            angular_momentum=angular_momentum,
            meson_radius=width_radius,
            energy_dependent=self.energy_dependent_width,
        )

        if self.form_factor:
            expression *= blatt_weisskopf_factor(
                mass=mass,
                mass0=mass0,
                daughter_mass1=daughter_mass1,
                daughter_mass2=daughter_mass2,
                angular_momentum=angular_momentum,
                meson_radius=radius,
            )

        parameters = {
            mass0: resonance.mass,
            gamma0: resonance.width,
        }
        if self.form_factor:
            parameters[radius] = self.meson_radius_default
        return expression, parameters
