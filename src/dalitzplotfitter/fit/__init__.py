"""Fit configuration and minimization."""

from .genfit import GaussianFitResult, GenFit, GenFitRecord, GenFitResult
from .minimizer import Minimizer, MultiStartResult
from .outliers import (
    OutlierSelection,
    RobustGaussianFitResult,
    genfit_distribution,
    genfit_outlier_selection,
    genfit_robust_gaussian_fit,
    genfit_robust_summary,
    robust_gaussian_fit,
    robust_outlier_mask,
)
from .parameters import Parameter, ParameterKind

__all__ = [
    "GaussianFitResult",
    "GenFit",
    "GenFitRecord",
    "GenFitResult",
    "Minimizer",
    "MultiStartResult",
    "OutlierSelection",
    "Parameter",
    "ParameterKind",
    "RobustGaussianFitResult",
    "genfit_distribution",
    "genfit_outlier_selection",
    "genfit_robust_gaussian_fit",
    "genfit_robust_summary",
    "robust_gaussian_fit",
    "robust_outlier_mask",
]
