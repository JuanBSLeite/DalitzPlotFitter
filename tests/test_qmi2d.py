import jax.numpy as jnp

from dalitzplotfitter import (
    DalitzAmplitude,
    DecayChannel,
    DecayModel,
    Parameter,
    QMI2D,
    RealImag,
    enable_x64,
    physical_bin_mask,
)


enable_x64()


def _data(s12, s13):
    return {"s12": jnp.asarray(s12), "s13": jnp.asarray(s13)}


def test_qmi2d_none_is_piecewise_constant_per_bin():
    model = QMI2D(
        s12_edges=(0.0, 1.0, 2.0),
        s13_edges=(0.0, 1.0, 2.0),
        magnitudes=((1.0, 2.0), (3.0, 4.0)),
        phases=((0.0, 0.1), (0.2, 0.3)),
        interpolation="none",
    )
    mag, phase = model.interpolated_magnitude_phase(
        _data([0.2, 0.2, 1.2, 1.2], [0.2, 1.2, 0.2, 1.2])
    )
    assert bool(jnp.allclose(mag, jnp.asarray([1.0, 2.0, 3.0, 4.0])))
    assert bool(jnp.allclose(phase, jnp.asarray([0.0, 0.1, 0.2, 0.3])))


def test_qmi2d_linear_interpolates_between_bin_centers():
    model = QMI2D(
        s12_edges=(0.0, 1.0, 2.0),
        s13_edges=(0.0, 1.0, 2.0),
        magnitudes=((1.0, 3.0), (3.0, 5.0)),
        phases=((0.0, 0.2), (0.2, 0.4)),
        interpolation="linear",
    )
    mag, phase = model.interpolated_magnitude_phase(_data([1.0], [1.0]))
    assert abs(float(mag[0]) - 3.0) < 1e-12
    assert abs(float(phase[0]) - 0.2) < 1e-12


def test_qmi2d_cubic_reproduces_all_bin_center_values():
    edges = (0.0, 1.0, 2.0, 3.0, 4.0)
    magnitudes = tuple(tuple(float(i + 2 * j + 1) for j in range(4)) for i in range(4))
    phases = tuple(tuple(float(0.1 * i - 0.05 * j) for j in range(4)) for i in range(4))
    model = QMI2D(
        s12_edges=edges,
        s13_edges=edges,
        magnitudes=magnitudes,
        phases=phases,
        interpolation="cubic",
    )
    centers = jnp.asarray([0.5, 1.5, 2.5, 3.5])
    xx, yy = jnp.meshgrid(centers, centers, indexing="ij")
    mag, phase = model.interpolated_magnitude_phase(_data(xx.ravel(), yy.ravel()))
    assert bool(jnp.allclose(mag.reshape(4, 4), jnp.asarray(magnitudes), atol=1e-12))
    assert bool(jnp.allclose(phase.reshape(4, 4), jnp.asarray(phases), atol=1e-12))


def test_qmi2d_folded_is_symmetric_under_s12_s13_exchange():
    model = QMI2D(
        s12_edges=(0.0, 1.0, 2.0),
        s13_edges=(0.0, 1.0, 2.0),
        magnitudes=((1.0, 2.0), (3.0, 4.0)),
        phases=((0.0, 0.2), (0.4, 0.6)),
        folded=True,
    )
    first = model(_data([0.2], [1.2]))
    second = model(_data([1.2], [0.2]))
    assert bool(jnp.allclose(first, second, atol=1e-12))


def test_physical_bin_mask_keeps_endpoint_bins_and_rejects_external_cells():
    channel = DecayChannel("D_s+", ("pi-", "pi+", "pi+"))
    m1, m2, m3 = channel.daughter_masses
    smin = (m1 + m2) ** 2
    smax = (channel.parent_mass - m3) ** 2
    edges = tuple(float(v) for v in jnp.linspace(smin, smax, 9))
    mask = physical_bin_mask(
        edges,
        edges,
        mother_mass=channel.parent_mass,
        masses=channel.daughter_masses,
        folded=True,
        samples_per_bin=257,
    )
    assert any(mask[-1])
    assert any(row[-1] for row in mask)
    assert not mask[-1][-1]


def test_qmi2d_active_mask_zeroes_inactive_piecewise_bins():
    model = QMI2D(
        s12_edges=(0.0, 1.0, 2.0),
        s13_edges=(0.0, 1.0, 2.0),
        magnitudes=((1.0, 2.0), (3.0, 4.0)),
        phases=((0.0, 0.1), (0.2, 0.3)),
        interpolation="none",
        active_mask=((True, True), (False, True)),
    )
    mag, phase = model.interpolated_magnitude_phase(_data([1.2], [0.2]))
    assert float(mag[0]) == 0.0
    assert float(phase[0]) == 0.0


def test_qmi2d_parameters_are_collected_and_change_decay_intensity():
    owner = "qmi2d"
    a00 = Parameter.dynamics("qmi2d.a00", 1.0, owner=owner, bounds=(0.0, None))
    p00 = Parameter.dynamics("qmi2d.p00", 0.0, owner=owner)
    field = QMI2D(
        s12_edges=(0.0, 2.0, 4.0),
        s13_edges=(0.0, 2.0, 4.0),
        magnitudes=((a00, 1.2), (0.8, 1.1)),
        phases=((p00, 0.2), (0.4, 0.6)),
        interpolation="linear",
        folded=True,
    )
    decay = DecayModel(
        DecayChannel("D_s+", ("pi-", "pi+", "pi+")),
        [DalitzAmplitude(owner, field, RealImag(1.0, 0.0))],
        normalization_resolution=30,
    )
    names = {parameter.name for parameter in decay.parameters}
    assert {"qmi2d.a00", "qmi2d.p00"}.issubset(names)

    data = decay.normalization_sample.as_dict()
    nominal = decay.intensity(data, {"qmi2d.a00": 1.0, "qmi2d.p00": 0.0})
    shifted = decay.intensity(data, {"qmi2d.a00": 1.8, "qmi2d.p00": 0.5})
    assert bool(jnp.any(jnp.abs(nominal - shifted) > 1e-10))


def test_qmi2d_rejects_unknown_interpolation_mode():
    try:
        QMI2D(
            s12_edges=(0.0, 1.0),
            s13_edges=(0.0, 1.0),
            magnitudes=((1.0,),),
            phases=((0.0,),),
            interpolation="spline",
        )
    except ValueError as exc:
        assert "interpolation" in str(exc)
    else:
        raise AssertionError("QMI2D accepted an unknown interpolation mode")
