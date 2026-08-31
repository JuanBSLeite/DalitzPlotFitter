"""Bias and pull-calibration diagnostics for GenFit pseudoexperiments."""

from __future__ import annotations

import math

import numpy as np


def genfit_bias_summary(result) -> list[dict[str, float | int | str]]:
    """Summarize fit bias and pull calibration for accepted pseudoexperiments.

    This diagnostic intentionally uses all accepted fits and does not perform
    distribution-level outlier rejection. That preserves the frequentist
    pseudoexperiment test: each toy fluctuates statistically around one fixed
    injected truth point, and the estimator is unbiased when the ensemble mean
    is compatible with that truth.
    """

    rows: list[dict[str, float | int | str]] = []
    for name in result.parameter_names:
        values = np.asarray(result.values(name), dtype=float)
        errors = np.asarray(result.errors(name), dtype=float)
        truth = float(result.truth_values[name])
        finite = np.isfinite(values) & np.isfinite(errors) & (errors > 0.0)
        values = values[finite]
        errors = errors[finite]
        n = int(values.size)

        if n:
            mean_fit = float(np.mean(values))
            bias = mean_fit - truth
        else:
            mean_fit = math.nan
            bias = math.nan

        if n > 1:
            sample_std = float(np.std(values, ddof=1))
            mean_error = sample_std / math.sqrt(n)
        else:
            sample_std = math.nan
            mean_error = math.nan

        bias_significance = (
            bias / mean_error
            if np.isfinite(mean_error) and mean_error > 0.0
            else math.nan
        )
        relative_bias = (
            bias / truth if np.isfinite(bias) and truth != 0.0 else math.nan
        )

        if n:
            pulls = (values - truth) / errors
            pull_mean = float(np.mean(pulls))
        else:
            pull_mean = math.nan
        pull_width = float(np.std(pulls, ddof=1)) if n > 1 else math.nan
        pull_mean_error = pull_width / math.sqrt(n) if n > 1 else math.nan
        pull_width_error = (
            pull_width / math.sqrt(2.0 * (n - 1)) if n > 1 else math.nan
        )

        rows.append(
            {
                "name": name,
                "entries": n,
                "truth": truth,
                "mean_fit": mean_fit,
                "sample_std": sample_std,
                "bias": bias,
                "relative_bias": relative_bias,
                "mean_error": mean_error,
                "bias_significance": bias_significance,
                "pull_mean": pull_mean,
                "pull_mean_error": pull_mean_error,
                "pull_width": pull_width,
                "pull_width_error": pull_width_error,
            }
        )
    return rows


def print_genfit_bias_summary(result) -> None:
    """Print the primary GenFit bias and pull-calibration table."""

    print(
        f"{'parameter':18s} {'truth':>11s} {'mean fit':>11s} "
        f"{'bias':>11s} {'err(mean)':>11s} {'bias/err':>10s} "
        f"{'pull mean':>11s} {'pull width':>11s}"
    )
    for row in genfit_bias_summary(result):
        print(
            f"{row['name']:18s} "
            f"{row['truth']:11.6g} "
            f"{row['mean_fit']:11.6g} "
            f"{row['bias']:11.6g} "
            f"{row['mean_error']:11.6g} "
            f"{row['bias_significance']:10.3f} "
            f"{row['pull_mean']:11.4f} "
            f"{row['pull_width']:11.4f}"
        )
