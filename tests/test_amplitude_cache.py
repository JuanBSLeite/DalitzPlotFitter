import jax.numpy as jnp

from dalitzplotfitter import RealImag, enable_x64
from dalitzplotfitter.amplitude import AmplitudeComponent, PreparedAmplitudeCache
from dalitzplotfitter.fit import Parameter
from dalitzplotfitter.integration import normalization_matrix


enable_x64()


class CountingAmplitude:
    def __init__(self, base):
        self.base = jnp.asarray(base, dtype=jnp.complex128)
        self.calls = 0

    def __call__(self, data, parameters=None):
        self.calls += 1
        scale = 1.0 if parameters is None else parameters.get("scale", 1.0)
        n = len(next(iter(data.values())))
        return jnp.resize(self.base, n) * scale


def _coefficient(prefix, x, y, *, fixed=False):
    return RealImag(
        Parameter.coefficient(f"{prefix}.x", x, fixed=fixed, owner=prefix),
        Parameter.coefficient(f"{prefix}.y", y, fixed=fixed, owner=prefix),
    )


def test_coefficient_only_fit_never_reevaluates_dynamics():
    f1 = CountingAmplitude([1.0 + 0.0j, 2.0 + 0.0j])
    f2 = CountingAmplitude([0.5 + 0.2j, -0.3 + 0.1j])
    c1 = _coefficient("a", 1.0, 0.0, fixed=True)
    c2 = _coefficient("b", 0.5, 0.2)
    components = (AmplitudeComponent("a", f1, c1), AmplitudeComponent("b", f2, c2))
    data = {"x": jnp.arange(8.0)}
    norm_data = {"x": jnp.arange(32.0)}
    cache = PreparedAmplitudeCache.prepare(
        components,
        data=data,
        normalization_data=norm_data,
        normalization_weights=jnp.ones(32),
        parameters=(*c1.parameters, *c2.parameters),
    )
    assert (f1.calls, f2.calls) == (2, 2)
    assert cache.is_compact
    assert cache.data is None
    assert cache.normalization_data is None
    assert cache.normalization_components is None
    intensity1, norm1 = cache.evaluate({"b.x": 0.4, "b.y": 0.1})
    intensity2, norm2 = cache.evaluate({"b.x": 0.8, "b.y": -0.5})
    assert (f1.calls, f2.calls) == (2, 2)
    assert not jnp.allclose(intensity1, intensity2)
    assert not jnp.allclose(norm1, norm2)


def test_only_component_with_floating_dynamics_is_reevaluated():
    f1 = CountingAmplitude([1.0 + 0.0j, 2.0 + 0.0j])
    f2 = CountingAmplitude([0.2 + 0.1j, 0.4 - 0.2j])
    c1 = _coefficient("a", 1.0, 0.0, fixed=True)
    c2 = _coefficient("b", 0.5, 0.3)
    dynamic = Parameter.dynamics(
        "a.scale", 1.0, backend_name="scale", owner="a"
    )
    components = (AmplitudeComponent("a", f1, c1), AmplitudeComponent("b", f2, c2))
    cache = PreparedAmplitudeCache.prepare(
        components,
        data={"x": jnp.arange(8.0)},
        normalization_data={"x": jnp.arange(32.0)},
        normalization_weights=jnp.ones(32),
        parameters=(*c1.parameters, *c2.parameters, dynamic),
    )
    assert not cache.is_compact
    assert (f1.calls, f2.calls) == (2, 2)
    cache.evaluate({"a.scale": 1.2, "b.x": 0.6, "b.y": 0.4})
    assert f1.calls == 4
    assert f2.calls == 2


def test_component_normalization_sets_matrix_diagonal_to_one():
    f1 = CountingAmplitude([1.0 + 0.0j, 2.0 + 0.0j])
    f2 = CountingAmplitude([0.5 + 0.2j, -0.3 + 0.1j])
    c1 = _coefficient("a", 1.0, 0.0, fixed=True)
    c2 = _coefficient("b", 0.5, 0.2)
    cache = PreparedAmplitudeCache.prepare(
        (AmplitudeComponent("a", f1, c1), AmplitudeComponent("b", f2, c2)),
        data={"x": jnp.arange(8.0)},
        normalization_data={"x": jnp.arange(32.0)},
        normalization_weights=jnp.ones(32),
        parameters=(*c1.parameters, *c2.parameters),
        normalize_components=True,
    )
    diagonal = jnp.real(jnp.diag(cache.normalization_matrix_fixed))
    assert jnp.allclose(diagonal, jnp.ones(2), rtol=1e-12, atol=1e-12)


def test_compact_efficiency_weighted_matrix_matches_direct_recomputation():
    f1 = CountingAmplitude([1.0 + 0.0j, 2.0 - 0.2j, 0.3 + 0.4j])
    f2 = CountingAmplitude([0.2 + 0.1j, 0.4 - 0.2j, -0.7 + 0.3j])
    c1 = _coefficient("a", 1.0, 0.0, fixed=True)
    c2 = _coefficient("b", 0.5, 0.3)
    weights = jnp.linspace(0.5, 1.5, 48)
    efficiency = jnp.linspace(0.7, 1.0, 48)
    components = (
        AmplitudeComponent("a", f1, c1),
        AmplitudeComponent("b", f2, c2),
    )
    norm_data = {"x": jnp.arange(48.0)}
    cache = PreparedAmplitudeCache.prepare(
        components,
        data={"x": jnp.arange(12.0)},
        normalization_data=norm_data,
        normalization_weights=weights,
        efficiency_normalization=efficiency,
        parameters=(*c1.parameters, *c2.parameters),
    )

    raw_norm = jnp.stack(
        [component.function(norm_data, None) for component in components], axis=1
    )
    bare = normalization_matrix(raw_norm, weights)
    scales = 1.0 / jnp.sqrt(jnp.real(jnp.diag(bare)))
    expected = normalization_matrix(raw_norm * scales, weights, efficiency)
    assert cache.is_compact
    assert jnp.allclose(
        cache.normalization_matrix_fixed,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_chunked_compact_normalization_matches_direct_partial_tail():
    f1 = CountingAmplitude([1.0 + 0.0j, 2.0 - 0.2j, 0.3 + 0.4j])
    f2 = CountingAmplitude([0.2 + 0.1j, 0.4 - 0.2j, -0.7 + 0.3j])
    c1 = _coefficient("a", 1.0, 0.0, fixed=True)
    c2 = _coefficient("b", 0.5, 0.3)
    weights = jnp.linspace(0.5, 1.5, 23)
    efficiency = jnp.linspace(0.65, 1.0, 23)
    components = (
        AmplitudeComponent("a", f1, c1),
        AmplitudeComponent("b", f2, c2),
    )
    norm_data = {"x": jnp.arange(23.0)}
    cache = PreparedAmplitudeCache.prepare(
        components,
        data={"x": jnp.arange(9.0)},
        normalization_data=norm_data,
        normalization_weights=weights,
        efficiency_normalization=efficiency,
        parameters=(*c1.parameters, *c2.parameters),
        normalization_chunk_size=7,
    )

    raw_norm = jnp.stack(
        [component.function(norm_data, None) for component in components], axis=1
    )
    bare = normalization_matrix(raw_norm, weights)
    scales = 1.0 / jnp.sqrt(jnp.real(jnp.diag(bare)))
    expected = normalization_matrix(raw_norm * scales, weights, efficiency)
    assert jnp.allclose(cache.component_scales, scales, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(
        cache.normalization_matrix_fixed,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_compact_unscaled_matrix_matches_direct_recomputation():
    f1 = CountingAmplitude([1.0 + 0.0j, 2.0 - 0.2j])
    f2 = CountingAmplitude([0.2 + 0.1j, 0.4 - 0.2j])
    c1 = _coefficient("a", 1.0, 0.0, fixed=True)
    c2 = _coefficient("b", 0.5, 0.3)
    weights = jnp.linspace(0.5, 1.5, 24)
    components = (
        AmplitudeComponent("a", f1, c1),
        AmplitudeComponent("b", f2, c2),
    )
    norm_data = {"x": jnp.arange(24.0)}
    cache = PreparedAmplitudeCache.prepare(
        components,
        data={"x": jnp.arange(8.0)},
        normalization_data=norm_data,
        normalization_weights=weights,
        parameters=(*c1.parameters, *c2.parameters),
        normalize_components=False,
    )
    raw_norm = jnp.stack(
        [component.function(norm_data, None) for component in components], axis=1
    )
    expected = normalization_matrix(raw_norm, weights)
    assert jnp.allclose(
        cache.normalization_matrix_fixed,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_floating_component_is_renormalized_after_dynamic_change():
    f1 = CountingAmplitude([1.0 + 0.0j, 2.0 + 0.0j])
    f2 = CountingAmplitude([0.2 + 0.1j, 0.4 - 0.2j])
    c1 = _coefficient("a", 1.0, 0.0, fixed=True)
    c2 = _coefficient("b", 0.5, 0.3)
    dynamic = Parameter.dynamics(
        "a.scale", 1.0, backend_name="scale", owner="a"
    )
    cache = PreparedAmplitudeCache.prepare(
        (AmplitudeComponent("a", f1, c1), AmplitudeComponent("b", f2, c2)),
        data={"x": jnp.arange(8.0)},
        normalization_data={"x": jnp.arange(32.0)},
        normalization_weights=jnp.ones(32),
        parameters=(*c1.parameters, *c2.parameters, dynamic),
        normalize_components=True,
    )
    _, norm_components = cache._evaluate_components({"a.scale": 4.0})
    matrix = cache._matrix_with_dynamic_blocks(norm_components)
    diagonal = jnp.real(jnp.diag(matrix))
    assert jnp.allclose(diagonal, jnp.ones(2), rtol=1e-12, atol=1e-12)


def test_multiple_dynamic_rows_match_full_matrix_recomputation():
    f1 = CountingAmplitude([1.0 + 0.0j, 2.0 - 0.2j, 0.3 + 0.4j])
    f2 = CountingAmplitude([0.2 + 0.1j, 0.4 - 0.2j, -0.7 + 0.3j])
    f3 = CountingAmplitude([0.8 - 0.4j, 0.1 + 0.6j, 0.5 + 0.2j])
    c1 = _coefficient("a", 1.0, 0.0)
    c2 = _coefficient("b", 0.5, 0.3)
    c3 = _coefficient("c", 0.2, -0.1, fixed=True)
    d1 = Parameter.dynamics("a.scale", 1.0, backend_name="scale", owner="a")
    d2 = Parameter.dynamics("b.scale", 1.0, backend_name="scale", owner="b")
    weights = jnp.linspace(0.5, 1.5, 48)
    efficiency = jnp.linspace(0.7, 1.0, 48)
    cache = PreparedAmplitudeCache.prepare(
        (
            AmplitudeComponent("a", f1, c1),
            AmplitudeComponent("b", f2, c2),
            AmplitudeComponent("c", f3, c3),
        ),
        data={"x": jnp.arange(12.0)},
        normalization_data={"x": jnp.arange(48.0)},
        normalization_weights=weights,
        efficiency_normalization=efficiency,
        parameters=(*c1.parameters, *c2.parameters, *c3.parameters, d1, d2),
    )
    _, norm_components = cache._evaluate_components(
        {"a.scale": 1.4, "b.scale": 0.8}
    )
    matrix = cache._matrix_with_dynamic_blocks(norm_components)
    expected = normalization_matrix(norm_components, weights, efficiency)
    assert jnp.allclose(matrix, expected, rtol=1e-12, atol=1e-12)
