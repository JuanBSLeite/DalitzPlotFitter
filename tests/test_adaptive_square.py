import jax.numpy as jnp
import numpy as np

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
    model = _ToyModel(center=1.5, scale=10.0)  # effectively smooth
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

    adaptive_area = jnp.mean(adaptive.sample.weights)
    uniform_area = jnp.mean(uniform.weights)
    assert jnp.allclose(adaptive_area, uniform_area, rtol=2e-3, atol=1e-6)


def test_adaptive_square_refines_narrow_structure_without_metadata():
    channel = DecayChannel("B+", ("K-", "K+", "K+"))
    center = 1.01946**2
    model = _ToyModel(center=center, scale=1.01946 * 0.00425)
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

    assert result.n_leaves > (10 * 2) ** 2
    assert int(result.leaf_depths.max()) >= 4

    # The deepest cells must cluster near the narrow feature in s12 even
    # though the adaptive algorithm was never given its center or width.
    deepest = result.leaf_depths == result.leaf_depths.max()
    x0, x1, y0, y1 = result.leaf_bounds[deepest].T
    mp = 0.5 * (x0 + x1)
    tp = 0.5 * (y0 + y1)
    from dalitzplotfitter import square_dalitz_to_invariants

    s12, _, _ = square_dalitz_to_invariants(
        jnp.asarray(mp),
        jnp.asarray(tp),
        mother_mass=channel.parent_mass,
        masses=channel.daughter_masses,
        pair=(0, 1),
    )
    distance = np.abs(np.asarray(s12) - center)
    assert np.quantile(distance, 0.5) < 0.03
