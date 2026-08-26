"""AmpForm model construction and numerical compilation."""

from __future__ import annotations

from dataclasses import dataclass

from .model import CompiledModel


@dataclass(frozen=True)
class AmplitudeBuilder:
    """Build an AmpForm symbolic model from a QRules reaction.

    ``resonance_dynamics`` controls who owns the resonance line shape:

    - ``"ampform"`` keeps AmpForm's standard relativistic Breit-Wigner;
    - ``"laura"`` inserts DalitzPlotFitter's Laura++-convention symbolic
      relativistic Breit-Wigner through AmpForm's custom-dynamics interface.

    AmpForm still owns topology, helicity/angular structure and symmetrization in
    both cases.
    """

    reaction: object
    use_default_dynamics: bool = True
    form_factor: bool = True
    energy_dependent_width: bool = True
    resonance_dynamics: str = "ampform"

    def build(self):
        import ampform

        builder = ampform.get_builder(self.reaction)
        if self.use_default_dynamics:
            from ampform.dynamics.builder import create_non_dynamic_with_ff

            initial_particle = self.reaction.initial_state[-1]
            builder.dynamics.assign(initial_particle, create_non_dynamic_with_ff)

            if self.resonance_dynamics == "ampform":
                from ampform.dynamics.builder import RelativisticBreitWignerBuilder

                resonance_builder = RelativisticBreitWignerBuilder(
                    form_factor=self.form_factor,
                    energy_dependent_width=self.energy_dependent_width,
                )
            elif self.resonance_dynamics == "laura":
                from dalitzplotfitter.dynamics import (
                    LauraRelativisticBreitWignerBuilder,
                )

                resonance_builder = LauraRelativisticBreitWignerBuilder(
                    form_factor=self.form_factor,
                    energy_dependent_width=self.energy_dependent_width,
                )
            else:
                raise ValueError(
                    "resonance_dynamics must be either 'ampform' or 'laura'"
                )

            for name in self.reaction.get_intermediate_particles().names:
                builder.dynamics.assign(name, resonance_builder)
        return builder.formulate()


def compile_model(model, *, use_cse: bool = True) -> CompiledModel:
    """Compile an AmpForm model to the project's numerical backend."""

    from tensorwaves.function.sympy import create_parametrized_function

    expression = getattr(model, "expression", None)
    if expression is None:
        expression = model.intensity
    if hasattr(expression, "doit"):
        expression = expression.doit()
    function = create_parametrized_function(
        expression=expression,
        parameters=model.parameter_defaults,
        backend="jax",
        use_cse=use_cse,
    )
    return CompiledModel(function)
