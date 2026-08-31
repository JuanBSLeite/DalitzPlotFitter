import jax.numpy as jnp

from dalitzplotfitter import AdaptiveSquareDalitzGrid, DecayChannel, SquareDalitzGrid, enable_x64
from dalitzplotfitter.amplitude import AmplitudeComponent, CoherentAmplitudeModel
from dalitzplotfitter.coefficients import RealImag


enable_x64()


class _InvariantAmplitude:
    def __init__(self, center, scale):
        self.center = float(center)
        self.scale = float(scale)

    def __call__(self, data, parameters=None):
        return 1.0 / (data["s12"] - self.center + 1j * self.scale)


class _ToyModel:
    def __init__(self, center, scale):
        component = AmplitudeComponent(
            "narrow",
            _InvariantAmplitude(center, scale),
            RealImag(1.0, 0.0),
        )
        self._amplitude_model = CoherentAmplitudeModel((component,))

    @property
    def amplitude_model(self):
        return self._amplitude_model


def test_adaptive_square_constant_measure_matches_uniform_area():
    channel = DecayChannel("B+", ("K-", "K+", "K+"))
    model = _ToyModel(center=1.5, scale=10.0)
    adaptive = AdaptiveSquareDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        pair=(0, 1),
        base_resolution=8,
        min_depth=1,
        max_depth=2,
        tolerance=0.2,
    ).build(model)
    uniform = SquareDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        pair=(0, 1),
        resolution=500,
        quadrature="midpoint",
    ).sample()

    assert jnp.allclose(
        jnp.mean(adaptive.sample.weights),
        jnp.mean(uniform.weights),
        rtol=2e-3,
        atol=1e-6,
    )


def test_adaptive_square_refines_a_narrow_structure():
    channel = DecayChannel("B+", ("K-", "K+", "K+"))
    model = _ToyModel(
        center=1.01946**2,
        scale=1.01946 * 0.00425,
    )
    result = AdaptiveSquareDalitzGrid(
        channel.parent_mass,
        channel.daughter_masses,
        pair=(0, 1),
        base_resolution=10,
        min_depth=1,
        max_depth=5,
        tolerance=0.03,
        matrix_floor=1e-10,
    ).build(model)

    # Fundamental regression only: a narrow numerical structure must trigger
    # non-trivial refinement beyond the forced first subdivision. Detailed
    # convergence/localisation studies belong in notebook 14/full validation.
    assert result.n_leaves > (10 * 2) ** 2
    assert int(result.leaf_depths.max()) >= 4
