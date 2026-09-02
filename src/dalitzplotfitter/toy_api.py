"""Public toy-generation API with optional ROOT persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from dalitzplotfitter.io import write_cp_phase_space_sample, write_phase_space_sample
from dalitzplotfitter.kinematics import PhaseSpaceSample
from dalitzplotfitter.toy_accept import (
    CPToyBackground,
    ToyBackground,
    generate_cp_toy as _generate_cp_toy_accept,
    generate_signal_toy as _generate_signal_toy_accept,
    generate_toy as _generate_toy_accept,
)
from dalitzplotfitter.toy_inverse import (
    PreparedInverseToyGenerator,
    generate_cp_toy_inverse,
    generate_signal_toy_inverse,
    generate_toy_inverse,
    prepare_inverse_toy_generator,
)

PathLike = str | Path
_PUBLIC_METHODS = ("accept-reject", "inverse-transform")


def _validate_method(method: str) -> None:
    if method not in _PUBLIC_METHODS:
        raise ValueError(f"method must be one of {_PUBLIC_METHODS}, got {method!r}")


def _validate_inverse_options(method: str, pool_size: int | None, batch_size: int | None) -> None:
    if method == "inverse-transform" and pool_size is not None:
        raise ValueError("pool_size applies only to method='accept-reject'")
    if method == "inverse-transform" and batch_size is not None:
        raise ValueError("batch_size applies only to method='accept-reject'")


def generate_signal_toy(
    model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    efficiency=None,
    veto=None,
    seed: int | None = None,
    pool_size: int | None = None,
    method: str = "inverse-transform",
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    inverse_resolution: int = 1024,
    inverse_quantile_resolution: int | None = None,
    output_root: PathLike | None = None,
    output_tree: str = "DecayTree",
    root_include_weights: bool = True,
    root_include_momenta: bool = True,
) -> PhaseSpaceSample:
    """Generate an unweighted signal toy.

    ``inverse-transform`` is the default sampler. ``accept-reject`` remains
    available explicitly as a reference and validation method.
    """

    _validate_method(method)
    _validate_inverse_options(method, pool_size, batch_size)
    if method == "inverse-transform":
        toy = generate_signal_toy_inverse(
            model,
            size,
            parameters=parameters,
            efficiency=efficiency,
            veto=veto,
            seed=seed,
            resolution=inverse_resolution,
            quantile_resolution=inverse_quantile_resolution,
        )
    else:
        toy = _generate_signal_toy_accept(
            model,
            size,
            parameters=parameters,
            efficiency=efficiency,
            veto=veto,
            seed=seed,
            pool_size=pool_size,
            batch_size=batch_size,
            envelope_safety=envelope_safety,
            max_restarts=max_restarts,
        )
    if output_root is not None:
        write_phase_space_sample(
            output_root,
            toy,
            tree=output_tree,
            include_weights=root_include_weights,
            include_momenta=root_include_momenta,
        )
    return toy


def generate_toy(
    model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    efficiency=None,
    veto=None,
    signal_fraction: float = 1.0,
    backgrounds: Sequence[ToyBackground] = (),
    seed: int | None = None,
    pool_size: int | None = None,
    shuffle: bool = True,
    method: str = "inverse-transform",
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    inverse_resolution: int = 1024,
    inverse_quantile_resolution: int | None = None,
    output_root: PathLike | None = None,
    output_tree: str = "DecayTree",
    root_include_weights: bool = True,
    root_include_momenta: bool = True,
) -> PhaseSpaceSample:
    """Generate signal/background pseudo-data.

    The default is ``method='inverse-transform'``. Use
    ``method='accept-reject'`` when an independent rejection-based reference
    sample is desired.
    """

    _validate_method(method)
    _validate_inverse_options(method, pool_size, batch_size)
    if method == "inverse-transform":
        toy = generate_toy_inverse(
            model,
            size,
            parameters=parameters,
            efficiency=efficiency,
            veto=veto,
            signal_fraction=signal_fraction,
            backgrounds=backgrounds,
            seed=seed,
            shuffle=shuffle,
            resolution=inverse_resolution,
            quantile_resolution=inverse_quantile_resolution,
        )
    else:
        toy = _generate_toy_accept(
            model,
            size,
            parameters=parameters,
            efficiency=efficiency,
            veto=veto,
            signal_fraction=signal_fraction,
            backgrounds=backgrounds,
            seed=seed,
            pool_size=pool_size,
            shuffle=shuffle,
            batch_size=batch_size,
            envelope_safety=envelope_safety,
            max_restarts=max_restarts,
        )
    if output_root is not None:
        write_phase_space_sample(
            output_root,
            toy,
            tree=output_tree,
            include_weights=root_include_weights,
            include_momenta=root_include_momenta,
        )
    return toy


def generate_cp_toy(
    plus_model,
    minus_model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    plus_efficiency=None,
    minus_efficiency=None,
    plus_veto=None,
    minus_veto=None,
    signal_fraction: float = 1.0,
    backgrounds: Sequence[CPToyBackground] = (),
    seed: int | None = None,
    pool_size: int | None = None,
    shuffle: bool = True,
    method: str = "inverse-transform",
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    inverse_resolution: int = 1024,
    inverse_quantile_resolution: int | None = None,
    output_root: PathLike | None = None,
    output_tree: str = "DecayTree",
    charge_branch: str = "charge",
    root_include_weights: bool = True,
    root_include_momenta: bool = True,
) -> tuple[PhaseSpaceSample, PhaseSpaceSample]:
    """Generate a CP toy and optionally save both charges in one ROOT TTree.

    ``inverse-transform`` is the default for both charge samples;
    ``accept-reject`` remains available explicitly.
    """

    _validate_method(method)
    _validate_inverse_options(method, pool_size, batch_size)
    if method == "inverse-transform":
        plus_toy, minus_toy = generate_cp_toy_inverse(
            plus_model,
            minus_model,
            size,
            parameters=parameters,
            plus_efficiency=plus_efficiency,
            minus_efficiency=minus_efficiency,
            plus_veto=plus_veto,
            minus_veto=minus_veto,
            signal_fraction=signal_fraction,
            backgrounds=backgrounds,
            seed=seed,
            shuffle=shuffle,
            resolution=inverse_resolution,
            quantile_resolution=inverse_quantile_resolution,
        )
    else:
        plus_toy, minus_toy = _generate_cp_toy_accept(
            plus_model,
            minus_model,
            size,
            parameters=parameters,
            plus_efficiency=plus_efficiency,
            minus_efficiency=minus_efficiency,
            plus_veto=plus_veto,
            minus_veto=minus_veto,
            signal_fraction=signal_fraction,
            backgrounds=backgrounds,
            seed=seed,
            pool_size=pool_size,
            shuffle=shuffle,
            batch_size=batch_size,
            envelope_safety=envelope_safety,
            max_restarts=max_restarts,
        )
    if output_root is not None:
        write_cp_phase_space_sample(
            output_root,
            plus_toy,
            minus_toy,
            tree=output_tree,
            charge_branch=charge_branch,
            include_weights=root_include_weights,
            include_momenta=root_include_momenta,
        )
    return plus_toy, minus_toy


__all__ = [
    "CPToyBackground",
    "PreparedInverseToyGenerator",
    "ToyBackground",
    "generate_cp_toy",
    "generate_signal_toy",
    "generate_toy",
    "prepare_inverse_toy_generator",
]
