import jax.numpy as jnp

from dalitzplotfitter.amplitude import AmplitudeComponent, PreparedAmplitudeCache
from dalitzplotfitter.coefficients import FitMagPhase
from dalitzplotfitter.fit import Parameter


class CountingAmplitude:
    def __init__(self, base):
        self.base = jnp.asarray(base, dtype=jnp.complex128)
        self.calls = 0

    def __call__(self, data, parameters=None):
        self.calls += 1
        scale = 1.0
        if parameters is not None:
            scale = parameters.get("scale", 1.0)
        n = len(next(iter(data.values())))
        return jnp.resize(self.base, n) * scale


def _fit_coefficient(prefix, r, phi, *, fixed=False):
    return FitMagPhase(
        r=Parameter.coefficient(f"{prefix}.r", r, fixed=fixed, owner=prefix),
        phi=Parameter.coefficient(f"{prefix}.phi", phi, fixed=fixed, owner=prefix),
    )


def test_coefficient_only_fit_never_reevaluates_dynamics():
    f1 = CountingAmplitude([1.0 + 0.0j, 2.0 + 0.0j])
    f2 = CountingAmplitude([0.5 + 0.2j, -0.3 + 0.1j])
    c1 = _fit_coefficient("a", 1.0, 0.0, fixed=True)
    c2 = _fit_coefficient("b", 0.5, 0.2)
    components = (
        AmplitudeComponent("a", f1, c1),
        AmplitudeComponent("b", f2, c2),
    )
    parameters = (*c1.parameters, *c2.parameters)
    data = {"x": jnp.arange(8.0)}
    norm_data = {"x": jnp.arange(32.0)}

    cache = PreparedAmplitudeCache.prepare(
        components,
        data=data,
        normalization_data=norm_data,
        normalization_weights=jnp.ones(32),
        parameters=parameters,
    )
    assert (f1.calls, f2.calls) == (2, 2)

    intensity1, norm1 = cache.evaluate({"b.r": 0.4, "b.phi": 0.1})
    intensity2, norm2 = cache.evaluate({"b.r": 0.8, "b.phi": -0.5})

    assert (f1.calls, f2.calls) == (2, 2)
    assert not jnp.allclose(intensity1, intensity2)
    assert not jnp.allclose(norm1, norm2)


def test_only_component_with_floating_dynamics_is_reevaluated():
    f1 = CountingAmplitude([1.0 + 0.0j, 2.0 + 0.0j])
    f2 = CountingAmplitude([0.2 + 0.1j, 0.4 - 0.2j])
    c1 = _fit_coefficient("a", 1.0, 0.0, fixed=True)
    c2 = _fit_coefficient("b", 0.5, 0.3)
    dynamic = Parameter.dynamics(
        "a.scale",
        1.0,
        backend_name="scale",
        owner="a",
    )
    components = (
        AmplitudeComponent("a", f1, c1),
        AmplitudeComponent("b", f2, c2),
    )
    data = {"x": jnp.arange(8.0)}
    norm_data = {"x": jnp.arange(32.0)}

    cache = PreparedAmplitudeCache.prepare(
        components,
        data=data,
        normalization_data=norm_data,
        normalization_weights=jnp.ones(32),
        parameters=(*c1.parameters, *c2.parameters, dynamic),
    )
    assert (f1.calls, f2.calls) == (2, 2)

    cache.evaluate({"a.scale": 1.2, "b.r": 0.6, "b.phi": 0.4})

    # The floating component is evaluated once on data and once on normalization MC.
    assert f1.calls == 4
    # The static component remains fully cached.
    assert f2.calls == 2
