import matplotlib.pyplot as plt
import numpy as np

from dalitzplotfitter import (
    DecayChannel,
    DecayModel,
    GenFit,
    NonResonant,
    Parameter,
    RealImag,
    Resonance,
    genfit_bias_summary,
    genfit_robust_summary,
    print_genfit_bias_summary,
    print_genfit_robust_summary,
    robust_gaussian_fit,
    robust_outlier_mask,
)


def _small_model():
    channel = DecayChannel("D+", ("pi-", "pi+", "pi+"))
    x = Parameter.coefficient("sigma.x", 0.6, owner="sigma", step=0.05)
    y = Parameter.coefficient("sigma.y", -0.4, owner="sigma", step=0.05)
    return DecayModel(
        channel,
        [
            Resonance(
                "rho770",
                (0, 1),
                RealImag(1.0, 0.0),
                mass=0.7693,
                width=0.1502,
                spin=1,
            ),
            Resonance(
                "sigma",
                (0, 1),
                RealImag(x, y),
                mass=0.478,
                width=0.324,
                spin=0,
            ),
            NonResonant(RealImag(0.2, 0.1)),
        ],
    )


def test_genfit_collects_parameter_and_fit_diagnostics():
    study = GenFit(
        _small_model(),
        n_fits=3,
        sample_size=250,
        grid_resolution=35,
        pool_size=2_500,
        seed=123,
        start_range=(-1.0, 1.0),
        ncall=5_000,
        verbose=0,
    )
    result = study.run()

    assert result.n_fits == 3
    assert result.values("sigma.x", valid_only=False).shape == (3,)
    assert result.errors("sigma.y", valid_only=False).shape == (3,)
    assert result.nll.shape == (3,)
    assert result.truth_nll.shape == (3,)
    assert result.edm.shape == (3,)
    assert result.nfcn.shape == (3,)
    assert result.converged_mask.shape == (3,)
    assert result.accepted_mask.shape == (3,)
    np.testing.assert_array_equal(result.valid_mask, result.accepted_mask)
    assert result.n_valid == result.n_accepted
    assert result.success_rate == result.acceptance_rate
    assert 0.0 <= result.convergence_rate <= 1.0
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.n_accepted <= result.n_converged

    for record in result.records:
        assert record.valid == record.accepted
        assert isinstance(record.rejection_reasons, tuple)
        if record.accepted:
            assert record.converged
            assert record.rejection_reasons == ()

    rows = result.summary()
    assert [row["name"] for row in rows] == ["sigma.x", "sigma.y", "nll"]
    assert all("gauss_mean" in row and "gauss_sigma" in row for row in rows)


def test_genfit_plot_returns_axis():
    study = GenFit(
        _small_model(),
        n_fits=2,
        sample_size=150,
        grid_resolution=25,
        pool_size=1_500,
        seed=456,
        start_range=(-1.0, 1.0),
        ncall=3_000,
        verbose=0,
    )
    result = study.run()
    ax = result.plot("sigma.x", bins=5)
    assert ax.get_xlabel() == "sigma.x"
    plt.close(ax.figure)


def test_genfit_is_reproducible_for_fixed_seeds():
    kwargs = dict(
        n_fits=2,
        sample_size=120,
        grid_resolution=20,
        pool_size=1_200,
        seed=789,
        pool_seed=42,
        start_range=(-0.8, 0.8),
        ncall=3_000,
        verbose=0,
    )
    first = GenFit(_small_model(), **kwargs).run()
    second = GenFit(_small_model(), **kwargs).run()
    np.testing.assert_allclose(
        first.values("sigma.x", valid_only=False),
        second.values("sigma.x", valid_only=False),
    )
    np.testing.assert_allclose(first.nll, second.nll)
    np.testing.assert_array_equal(first.converged_mask, second.converged_mask)
    np.testing.assert_array_equal(first.accepted_mask, second.accepted_mask)


def test_genfit_bias_summary_reports_bias_and_pull_calibration(capsys):
    result = GenFit(
        _small_model(),
        n_fits=4,
        sample_size=180,
        grid_resolution=25,
        pool_size=1_800,
        seed=246,
        start_range=(-1.0, 1.0),
        ncall=4_000,
        verbose=0,
    ).run()
    rows = genfit_bias_summary(result)
    assert [row["name"] for row in rows] == ["sigma.x", "sigma.y"]
    for row in rows:
        assert row["entries"] == result.n_accepted
        assert "bias" in row
        assert "mean_error" in row
        assert "bias_significance" in row
        assert "pull_mean" in row
        assert "pull_width" in row

    print_genfit_bias_summary(result)
    output = capsys.readouterr().out
    assert "bias/err" in output
    assert "pull mean" in output
    assert "pull width" in output


def test_robust_outlier_mask_removes_catastrophic_tail():
    values = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2, 25.0])
    selection = robust_outlier_mask(values, threshold=5.0)
    assert selection.n_outliers == 1
    assert not selection.mask[-1]
    assert selection.n_kept == 5


def test_robust_gaussian_fit_reports_rejected_entries():
    values = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2, 50.0])
    fit = robust_gaussian_fit(values, threshold=5.0)
    assert fit.n_entries == 6
    assert fit.n_kept == 5
    assert fit.n_outliers == 1
    assert np.isfinite(fit.mean)
    assert np.isfinite(fit.sigma)


def test_genfit_robust_summary_reports_outlier_columns():
    study = GenFit(
        _small_model(),
        n_fits=3,
        sample_size=150,
        grid_resolution=25,
        pool_size=1_500,
        seed=321,
        start_range=(-1.0, 1.0),
        ncall=3_000,
        verbose=0,
    )
    result = study.run()
    rows = genfit_robust_summary(result, threshold=5.0)
    assert [row["name"] for row in rows] == ["sigma.x", "sigma.y", "nll"]
    assert all("n_outliers" in row for row in rows)
    assert all("outlier_fraction" in row for row in rows)


def test_print_genfit_robust_summary_has_rejected_columns(capsys):
    study = GenFit(
        _small_model(),
        n_fits=2,
        sample_size=120,
        grid_resolution=20,
        pool_size=1_200,
        seed=987,
        start_range=(-0.8, 0.8),
        ncall=3_000,
        verbose=0,
    )
    result = study.run()
    print_genfit_robust_summary(result)
    output = capsys.readouterr().out
    assert "rejected" in output
    assert "rejected %" in output
