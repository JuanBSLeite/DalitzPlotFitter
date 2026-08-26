import jax
import jax.numpy as jnp

from dalitzplotfitter import ThreeBodyPhaseSpace, enable_x64
from dalitzplotfitter.toy import ToyGenerator


enable_x64()


def _energy_transform(momentum):
    return {"energy": momentum["p0"][:, 0]}


def _dalitz_transform(momentum):
    p1 = momentum["p0"]
    p2 = momentum["p1"]
    p3 = momentum["p2"]

    def mass_squared(p):
        return p[:, 0] ** 2 - jnp.sum(p[:, 1:] ** 2, axis=1)

    return {
        "s12": mass_squared(p1 + p2),
        "s23": mass_squared(p2 + p3),
    }


def _constant_intensity(data, parameters):
    del parameters
    return jnp.ones_like(next(iter(data.values())))


def test_accept_reject_generates_unweighted_continuous_phase_space():
    phase_space = ThreeBodyPhaseSpace(
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
    )
    generator = ToyGenerator(
        phase_space=phase_space,
        transformer=_energy_transform,
        pool_size=10_000,
        batch_size=4_000,
        envelope_safety=1.3,
    )

    toy, data = generator.generate(
        jax.random.key(2026),
        size=2_000,
        intensity=_constant_intensity,
        parameters={},
    )

    assert toy.size == 2_000
    assert data["energy"].shape == (2_000,)
    assert jnp.allclose(toy.weights, jnp.ones(2_000))
    assert bool(jnp.all(jnp.isfinite(toy.s12)))
    assert bool(jnp.all(jnp.isfinite(toy.s13)))
    assert bool(jnp.all(jnp.isfinite(toy.s23)))

    # Accept-reject generates fresh continuous candidates rather than drawing
    # repeatedly from a finite categorical pool. Exact duplicate s12 values
    # should therefore be absent for this deterministic test seed.
    assert int(jnp.unique(toy.s12).shape[0]) == toy.size


def test_accept_reject_includes_phase_space_proposal_weight():
    phase_space = ThreeBodyPhaseSpace(
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
    )
    generator = ToyGenerator(
        phase_space=phase_space,
        transformer=_energy_transform,
        pool_size=20_000,
        batch_size=5_000,
        envelope_safety=1.3,
    )

    toy, _ = generator.generate(
        jax.random.key(7),
        size=5_000,
        intensity=_constant_intensity,
        parameters={},
    )

    # For constant matrix element the physical Dalitz density is uniform in
    # ds12 ds23. The native proposal is not uniform in that measure, so its
    # Jacobian weights are required. Compare the unweighted toy mean with an
    # independent weighted proposal estimate of the same phase-space mean.
    reference = phase_space.generate(
        jax.random.key(8),
        size=100_000,
        with_momenta=False,
    )
    expected_mean = jnp.sum(reference.weights * reference.s12) / jnp.sum(
        reference.weights
    )
    assert jnp.isclose(jnp.mean(toy.s12), expected_mean, rtol=0.0, atol=0.025)


def test_deterministic_envelope_search_finds_narrow_peak():
    phase_space = ThreeBodyPhaseSpace(
        mother_mass=1.86966,
        masses=(0.13957, 0.13957, 0.13957),
    )
    target_u = jnp.asarray([[0.371, 0.613]])
    target = phase_space.from_unit_square(target_u, with_momenta=False)
    target_s12 = target.s12[0]
    target_s23 = target.s23[0]

    # Build a deliberately narrow synthetic peak. A coarse random pilot can
    # easily miss this structure, while the hierarchical deterministic search
    # must locate it reproducibly.
    s12_edges = phase_space.from_unit_square(
        jnp.asarray([[1.0, 0.5], [0.0, 0.5]]),
        with_momenta=False,
    )
    sigma12 = 0.0025 * float(jnp.ptp(s12_edges.s12))
    # The local s23 span depends on s12. Use the span at the target u1.
    target_edge = phase_space.from_unit_square(
        jnp.asarray([[target_u[0, 0], 0.0], [target_u[0, 0], 1.0]]),
        with_momenta=False,
    )
    sigma23 = 0.0025 * float(jnp.abs(target_edge.s23[1] - target_edge.s23[0]))

    def narrow_intensity(data, parameters):
        del parameters
        return jnp.exp(
            -0.5 * ((data["s12"] - target_s12) / sigma12) ** 2
            -0.5 * ((data["s23"] - target_s23) / sigma23) ** 2
        )

    generator = ToyGenerator(
        phase_space=phase_space,
        transformer=_dalitz_transform,
        envelope_grid_size=48,
        envelope_refinement_size=17,
        envelope_refinement_levels=5,
        envelope_top_k=8,
    )
    maximum, maximum_point = generator.estimate_maximum(narrow_intensity, {})

    assert maximum > 0.0
    assert jnp.allclose(maximum_point, target_u[0], rtol=0.0, atol=0.004)
