"""Transform four-momenta into AmpForm kinematic variables."""

from __future__ import annotations

from dataclasses import dataclass

from dalitzplotfitter.kinematics import PhaseSpaceSample


def _create_kinematic_expressions(model: object):
    """Create all kinematic expressions required by an AmpForm model.

    For identical final-state particles, AmpForm symmetrizes the amplitude by
    permuting the registered topologies. Recreate the same permutations in the
    kinematic adapter so variables such as ``m_02`` and ``theta_0^02`` are
    available to the numerical model.
    """

    from ampform.kinematics import HelicityAdapter

    adapter = HelicityAdapter(model.reaction_info)
    adapter.permutate_registered_topologies()
    return adapter.create_expressions()


@dataclass(frozen=True)
class KinematicTransformer:
    """JAX-backed transformer for an AmpForm model's kinematic variables."""

    model: object
    use_cse: bool = True

    def build(self):
        from tensorwaves.data import SympyDataTransformer

        expressions = _create_kinematic_expressions(self.model)
        return SympyDataTransformer.from_sympy(
            expressions,
            backend="jax",
            use_cse=self.use_cse,
        )

    def transform(self, sample: PhaseSpaceSample):
        transformer = self.build()
        return transformer(sample.as_momentum_dict())


def create_kinematic_transformer(model, *, use_cse: bool = True):
    """Create a TensorWaves kinematic transformer for an AmpForm model."""

    return KinematicTransformer(model=model, use_cse=use_cse).build()
