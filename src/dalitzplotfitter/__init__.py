"""DalitzPlotFitter public API."""

# JAX reads GPU allocator environment variables when its backend initializes.
# Set the project default before importing any submodule that imports JAX. Using
# setdefault keeps an explicit user/shell configuration authoritative.
import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from .amplitude import (
    AmplitudeComponent,
    CoherentAmplitudeModel,
    ConstantAmplitude,
    PreparedAmplitudeCache,
)
from .background import BackgroundCategory, CPBackgroundCategory
from .coefficients import CPRealImag, RealImag
from .config import enable_x64
from .constraints import ConstrainedNLL, GaussianConstraint
from .cp_workflow import CPBackgroundSpec, CPFitSession
from .decay import DalitzAmplitude, DecayChannel, DecayModel, NonResonant, Resonance
from .discriminants import (
    BreitWigner1D,
    Exponential1D,
    FactorizedDensity,
    Gaussian1D,
    Histogram1D,
    LineshapeIntensity1D,
)
from .dynamics import (
    LASS,
    QMI,
    QMI2D,
    BaBarFlatte,
    CovariantAngular,
    Flatte,
    GounarisSakurai,
    KMatrix,
    Pole,
    RelativisticBreitWigner,
    ResonanceAmplitude,
    ResonanceContext,
    ZemachP,
    ZemachPstar,
    Zemach_P,
    Zemach_Pstar,
    physical_bin_mask,
)
from .fit import Minimizer, MultiStartResult, Parameter, ParameterKind
from .integration import DalitzGaussLegendreGrid
from .io import (
    histogram_background_from_root,
    histogram_efficiency_from_root,
    read_phase_space_sample,
    read_root_histogram2d,
    read_root_tree,
    square_dalitz_background_from_root,
    square_dalitz_efficiency_from_root,
    write_cp_phase_space_sample,
    write_phase_space_sample,
    write_phase_space_samples,
)
from .kinematics import (
    CovariantKinematics,
    PhaseSpaceMC,
    PhaseSpaceSample,
    SquareDalitzGrid,
    boost_to_rest_frame,
    covariant_kinematics,
    covariant_kinematics_from_invariants,
    dalitz_s13_limits,
    invariants_to_square_dalitz,
    square_dalitz_jacobian,
    square_dalitz_to_invariants,
)
from .likelihood import MultiBackgroundNLL
from .pdf import SCFSignalPDF, SignalPDF
from .plotting import binned_data, plot_binned_data, plot_dalitz, plot_square_dalitz
from .resolution import (
    ConvolvedPDF1D,
    GaussianResolution1D,
    SparseMigration,
    SquareDalitzSCFMap,
)
from .sampling import weighted_resample
from .square_histograms import (
    SquareDalitzHistogramBackground,
    SquareDalitzHistogramEfficiency,
)
from .toy_api import (
    CPToyBackground,
    PreparedInverseToyGenerator,
    ToyBackground,
    generate_cp_toy,
    generate_signal_toy,
    generate_toy,
    prepare_inverse_toy_generator,
)
from .veto import (
    CompositeVeto,
    FunctionalVeto,
    MassWindowVeto,
    VetoMap,
    VetoedDensity,
    vetoed_signal_pdf,
)
from .workflow import BackgroundSpec, FitSession

__all__ = [
    "AmplitudeComponent", "BackgroundCategory", "BackgroundSpec", "BreitWigner1D", "CPBackgroundCategory",
    "CPBackgroundSpec", "CPFitSession", "CPToyBackground", "BaBarFlatte", "CPRealImag",
    "CoherentAmplitudeModel", "CompositeVeto", "ConstrainedNLL", "ConstantAmplitude",
    "ConvolvedPDF1D", "CovariantAngular", "CovariantKinematics", "DalitzAmplitude",
    "DecayChannel", "DecayModel", "Exponential1D", "FactorizedDensity", "FitSession",
    "Flatte", "FunctionalVeto", "Gaussian1D", "GaussianConstraint", "GaussianResolution1D",
    "GounarisSakurai", "Histogram1D", "KMatrix", "LASS", "LineshapeIntensity1D", "DalitzGaussLegendreGrid",
    "MassWindowVeto", "Minimizer", "MultiBackgroundNLL", "MultiStartResult", "NonResonant",
    "Parameter", "ParameterKind", "PhaseSpaceMC", "PhaseSpaceSample", "Pole",
    "PreparedAmplitudeCache", "PreparedInverseToyGenerator", "QMI", "QMI2D", "RealImag", "RelativisticBreitWigner",
    "Resonance", "ResonanceAmplitude", "ResonanceContext", "SCFSignalPDF", "SignalPDF",
    "SparseMigration", "SquareDalitzGrid", "SquareDalitzHistogramBackground", "SquareDalitzHistogramEfficiency",
    "SquareDalitzSCFMap", "ToyBackground", "VetoMap", "VetoedDensity", "ZemachP", "ZemachPstar",
    "Zemach_P", "Zemach_Pstar",
    "boost_to_rest_frame", "binned_data", "covariant_kinematics",
    "covariant_kinematics_from_invariants", "dalitz_s13_limits", "enable_x64",
    "generate_cp_toy", "generate_signal_toy", "generate_toy", "histogram_background_from_root",
    "histogram_efficiency_from_root", "invariants_to_square_dalitz", "physical_bin_mask",
    "plot_binned_data", "plot_dalitz", "plot_square_dalitz", "prepare_inverse_toy_generator",
    "read_phase_space_sample", "read_root_histogram2d", "read_root_tree",
    "square_dalitz_background_from_root", "square_dalitz_efficiency_from_root", "square_dalitz_jacobian",
    "square_dalitz_to_invariants", "vetoed_signal_pdf", "weighted_resample",
    "write_cp_phase_space_sample", "write_phase_space_sample", "write_phase_space_samples",
]
