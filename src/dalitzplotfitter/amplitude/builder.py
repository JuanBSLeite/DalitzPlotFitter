"""AmpForm model construction and numerical compilation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AmplitudeBuilder:
    """Build an AmpForm symbolic model from a QRules reaction."""

    reaction: object

    def build(self):
        import ampform

        builder = ampform.get_builder(self.reaction)
        return builder.formulate()


def compile_model(model, *, use_cse: bool = True):
    """Compile an AmpForm model to the project's numerical backend."""

    from tensorwaves.function.sympy import create_parametrized_function

    expression = getattr(model, "expression", None)
    if expression is None:
        expression = model.intensity
    if hasattr(expression, "doit"):
        expression = expression.doit()
    return create_parametrized_function(
        expression=expression,
        parameters=model.parameter_defaults,
        backend="jax",
        use_cse=use_cse,
    )
