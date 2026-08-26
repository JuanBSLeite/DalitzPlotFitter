"""Transform four-momenta into AmpForm kinematic variables."""

from __future__ import annotations

from dataclasses import dataclass

from dalitzplotfitter.kinematics import PhaseSpaceSample


@dataclass(frozen=True)
class KinematicTransformer:
    """JAX-backed transformer for an AmpForm model's kinematic variables."""

    model: object
    use_cse: bool = True

    def build(self):
        from tensorwaves.data import SympyDataTransformer

        return SympyDataTransformer.from_sympy(
            self.model.kinematic_variables,
            backend="jax",
            use_cse=self.use_cse,
        )

    def transform(self, sample: PhaseSpaceSample):
        transformer = self.build()
        return transformer(sample.as_momentum_dict())


def create_kinematic_transformer(model, *, use_cse: bool = True):
    """Create a TensorWaves kinematic transformer for an AmpForm model."""

    return KinematicTransformer(model=model, use_cse=use_cse).build()
