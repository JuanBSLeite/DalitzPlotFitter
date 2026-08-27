import math

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
    assert float(scan.best.fval) <= min(float(result.fval) for result in scan.valid_results) + 1e-8
