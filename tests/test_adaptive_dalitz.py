import jax.numpy as jnp
import numpy as np

from dalitzplotfitter import (
    AdaptiveDalitzGrid,
    DalitzGrid,
    DecayChannel,
    DecayModel,
    NonResonant,
    RealImag,
    Resonance,
    enable_x64,
)


enable_x64()


def _raw_matrix(model, sample):
    cache = model.prepare_cache(
        sample,
        normalization_sample=sample,
        normalize_components=False,
    )
    return np.asarray(cache.normalization_matrix_fixed)


def test_adaptive_dalitz_preserves_physical_area():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    model = DecayModel(
        channel,
        [NonResonant(RealImag(1.0, 0.0), name="NR")],
        normalize_components=False,
    )
    adaptive = AdaptiveDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        base_resolution=8,
        min_depth=1,
        max_depth=2,
        tolerance=0.1,
    ).build(model)
    reference = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=300,
    ).sample()

    assert jnp.allclose(
        jnp.mean(adaptive.sample.weights),
        jnp.mean(reference.weights),
        rtol=2e-5,
        atol=1e-7,
    )


def test_adaptive_dalitz_refines_narrow_phi_without_pole_metadata():
    channel = DecayChannel("B+", ("K-", "K+", "K+"))
    model = DecayModel(
        channel,
        [
            Resonance(
                "phi1020",
                (0, 1),
                RealImag(1.0, 0.0),
                mass=1.019461,
                width=0.004249,
                spin=1,
                resonance_radius=1.5,
                parent_radius=5.0,
            ),
            NonResonant(RealImag(0.3, 0.1), name="NR"),
        ],
        normalize_components=False,
    )

    adaptive = AdaptiveDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        base_resolution=10,
        min_depth=1,
        max_depth=4,
        tolerance=0.05,
        matrix_floor=1e-9,
        max_cells=80_000,
    ).build(model)

    assert adaptive.n_leaves > (10 * 2) ** 2
    assert int(adaptive.leaf_depths.max()) >= 2

    reference = DalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        resolution=350,
    ).sample()
    m_ref = _raw_matrix(model, reference)
    m_adapt = _raw_matrix(model, adaptive.sample)

    # Use the matrix's dominant physical scale rather than a relative error for
    # each element separately. Interference terms can be close to zero, where a
    # harmless absolute difference would otherwise produce an arbitrarily large
    # relative error.
    scale = np.max(np.abs(m_ref))
    matrix_error = np.max(np.abs(m_adapt - m_ref)) / scale
    assert float(matrix_error) < 1e-1
