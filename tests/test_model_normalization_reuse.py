import jax.numpy as jnp

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    NonResonant,
    RealImag,
    Resonance,
    enable_x64,
)


enable_x64()


def test_coefficient_only_model_reuses_fixed_normalization_across_datasets():
    model = DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [
            Resonance(
                "rho_test",
                pair=(0, 1),
                coefficient=RealImag(1.0, 0.0),
                mass=0.775,
                width=0.149,
                spin=1,
            ),
            NonResonant(RealImag(0.3, -0.2)),
        ],
        normalization_method="square-dalitz",
        normalization_resolution=35,
    )

    first_data = model.generate_phase_space(64, seed=901)
    second_data = model.generate_phase_space(64, seed=902)

    first = model.prepare_cache(first_data)
    assert len(model._fixed_normalization_templates) == 1
    template_scales, template_matrix = model._fixed_normalization_templates[True]

    second = model.prepare_cache(second_data)
    assert len(model._fixed_normalization_templates) == 1
    assert second.is_compact
    assert jnp.allclose(second.component_scales, template_scales, rtol=0.0, atol=0.0)
    assert jnp.allclose(
        second.normalization_matrix_fixed,
        template_matrix,
        rtol=0.0,
        atol=0.0,
    )
    assert jnp.allclose(
        first.normalization_matrix_fixed,
        second.normalization_matrix_fixed,
        rtol=0.0,
        atol=0.0,
    )
    assert not jnp.allclose(first.data_components, second.data_components)
    assert len(model._compact_data_kernels) == 1


def test_efficiency_weighted_cache_does_not_reuse_unweighted_matrix():
    model = DecayModel(
        DecayChannel("D+", ("pi-", "pi+", "pi+")),
        [NonResonant(RealImag(1.0, 0.0))],
        normalization_method="square-dalitz",
        normalization_resolution=25,
    )
    data = model.generate_phase_space(32, seed=903)
    unweighted = model.prepare_cache(data)
    efficiency = jnp.linspace(0.5, 1.0, model.normalization_sample.size)
    weighted = model.prepare_cache(data, efficiency_normalization=efficiency)

    assert len(model._fixed_normalization_templates) == 1
    assert not jnp.allclose(
        weighted.normalization_matrix_fixed,
        unweighted.normalization_matrix_fixed,
    )
