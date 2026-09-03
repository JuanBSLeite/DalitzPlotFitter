import math

import numpy as np
import pytest

from dalitzplotfitter import Minimizer, Parameter, enable_x64


enable_x64()


def test_minimizer_recovers_quadratic_minimum_from_displaced_start():
    parameters = (
        Parameter("x", -3.0, bounds=(-5.0, 5.0), step=0.1),
        Parameter("y", 4.0, bounds=(-5.0, 5.0), step=0.1),
    )

    def objective(values):
        return 0.5 * ((values["x"] - 1.25) / 0.3) ** 2 + 0.5 * (
            (values["y"] + 0.75) / 0.5
        ) ** 2

    result = Minimizer(objective, parameters).fit()
    assert result.valid
    assert math.isclose(float(result.values["x"]), 1.25, abs_tol=1e-8)
    assert math.isclose(float(result.values["y"]), -0.75, abs_tol=1e-8)


def test_minimizer_exposes_strategy_and_optional_hesse():
    parameter = Parameter("x", -2.0, bounds=(-5.0, 5.0), step=0.1)

    def objective(values):
        return (values["x"] - 1.5) ** 2

    result = Minimizer(objective, (parameter,)).fit(strategy=1, hesse=False)
    assert result.strategy == 1
    # A single strategy-1 MIGRAD without HESSE is intentionally a lightweight
    # fit path. Check that it reaches the expected minimum without requiring the
    # tighter numerical agreement of the default strategy-2 refinement.
    assert math.isclose(float(result.values["x"]), 1.5, abs_tol=2e-5)


@pytest.mark.parametrize("strategy", [-1, 3, True, 1.5])
def test_minimizer_rejects_invalid_strategy(strategy):
    parameter = Parameter("x", 0.0)

    def objective(values):
        return values["x"] ** 2

    with pytest.raises(ValueError, match="strategy"):
        Minimizer(objective, (parameter,)).fit(strategy=strategy)


def test_minimizer_reuses_compiled_backend_and_shared_value_gradient_point():
    parameter = Parameter("x", 0.0, bounds=(-5.0, 5.0))

    def objective(values):
        return (values["x"] - 2.0) ** 2

    minimizer = Minimizer(objective, (parameter,))
    backend1 = minimizer._backend()
    backend2 = minimizer._backend()
    assert backend1 is backend2

    _, _, fcn, grad = backend1
    assert math.isclose(fcn(1.5), 0.25, abs_tol=1e-12)
    assert np.allclose(grad(1.5), np.asarray([-1.0]), rtol=0.0, atol=1e-12)
    assert math.isclose(fcn(1.5), 0.25, abs_tol=1e-12)


def test_minimizer_reuses_backend_across_instances_for_same_objective():
    parameter = Parameter("x", 0.0, bounds=(-5.0, 5.0))

    def objective(values):
        return (values["x"] - 1.0) ** 2

    first = Minimizer(objective, (parameter,))
    second = Minimizer(objective, (parameter,), tolerance=1e-6, verbose=1)
    assert first._backend() is second._backend()


def test_minimizer_does_not_share_backend_when_fixed_value_changes():
    def objective(values):
        return (values["x"] - values["offset"]) ** 2

    first = Minimizer(
        objective,
        (
            Parameter("x", 0.0, bounds=(-5.0, 5.0)),
            Parameter("offset", 1.0, fixed=True),
        ),
    )
    second = Minimizer(
        objective,
        (
            Parameter("x", 0.0, bounds=(-5.0, 5.0)),
            Parameter("offset", 2.0, fixed=True),
        ),
    )
    assert first._backend() is not second._backend()


def test_multistart_selects_global_minimum_of_multimodal_objective():
    parameters = (
        Parameter("x", -1.4, bounds=(-2.0, 2.0), step=0.05),
    )

    def objective(values):
        x = values["x"]
        return (x**2 - 1.0) ** 2 + 0.08 * (x - 1.0) ** 2

    scan = Minimizer(objective, parameters).fit_multistart(
        n_starts=12,
        seed=731,
        include_default=True,
    )
    assert scan.best.valid
    assert len(scan.results) == 12
    assert len(scan.valid_results) >= 1
    assert float(scan.best.fval) < 1e-10
    assert math.isclose(float(scan.best.values["x"]), 1.0, abs_tol=1e-5)
    assert float(scan.best.fval) <= min(
        float(result.fval) for result in scan.valid_results
    ) + 1e-8


def test_multistart_trial_does_not_depend_on_number_of_later_starts():
    """A Minuit trial must not inherit state from other multistart trials.

    For a fixed seed, the random-start sequence is prefix-stable. Therefore the
    first preliminary trial in scans with 1, 2 and 10 starts is identical. Its
    fitted result must also be identical. If it changes, some state is leaking
    between the Minuit driver and the objective/PDF.
    """

    parameters = (
        Parameter("x", 0.0, bounds=(-3.0, 3.0), step=0.02),
        Parameter("y", 0.0, bounds=(-3.0, 3.0), step=0.02),
        Parameter("z", 0.0, bounds=(-3.0, 3.0), step=0.02),
    )

    def objective(values):
        x, y, z = values["x"], values["y"], values["z"]
        return (
            0.5 * ((x - 0.7) / 0.4) ** 2
            + 0.5 * ((y + 1.1) / 0.6) ** 2
            + 0.5 * ((z - 0.2) / 0.3) ** 2
            + 0.07 * x * y
        )

    minimizer = Minimizer(objective, parameters)
    scans = tuple(
        minimizer.fit_multistart(
            n_starts=n_starts,
            seed=314159,
            include_default=False,
            simplex=False,
        )
        for n_starts in (1, 2, 10)
    )

    reference_start = scans[0].starts[0]
    reference_result = scans[0].results[0]
    for scan in scans[1:]:
        assert scan.starts[0] == reference_start
        result = scan.results[0]
        assert bool(result.valid) == bool(reference_result.valid)
        assert math.isclose(
            float(result.fval),
            float(reference_result.fval),
            abs_tol=1e-12,
        )
        for name in ("x", "y", "z"):
            assert math.isclose(
                float(result.values[name]),
                float(reference_result.values[name]),
                abs_tol=1e-12,
            )