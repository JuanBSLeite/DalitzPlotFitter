"""Public toy-generation API with optional ROOT persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from dalitzplotfitter.io import write_cp_phase_space_sample, write_phase_space_sample
from dalitzplotfitter.kinematics import PhaseSpaceSample
from dalitzplotfitter.toy import (
    CPToyBackground,
    ToyBackground,
    generate_cp_toy as _generate_cp_toy,
    generate_signal_toy as _generate_signal_toy,
    generate_toy as _generate_toy,
)

PathLike = str | Path


def generate_signal_toy(
    model,
    size: int,
    *,
    parameters: Mapping[str, object] | None = None,
    efficiency=None,
    veto=None,
    seed: int | None = None,
    pool_size: int | None = None,
    method: str = "accept-reject",
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    output_root: PathLike | None = None,
    output_tree: str = "DecayTree",
    root_include_weights: bool = True,
    root_include_momenta: bool = True,
) -> PhaseSpaceSample:
    """Generate a signal toy and optionally persist it to one ROOT TTree."""

    toy = _generate_signal_toy(
        model,
        size,
        parameters=parameters,
        efficiency=efficiency,
        veto=veto,
        seed=seed,
        pool_size=pool_size,
        method=method,
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
    method: str = "accept-reject",
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    output_root: PathLike | None = None,
    output_tree: str = "DecayTree",
    root_include_weights: bool = True,
    root_include_momenta: bool = True,
) -> PhaseSpaceSample:
    """Generate a toy and optionally persist it to one ROOT TTree with uproot."""

    toy = _generate_toy(
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
        method=method,
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
    method: str = "accept-reject",
    batch_size: int | None = None,
    envelope_safety: float = 1.20,
    max_restarts: int = 10,
    output_root: PathLike | None = None,
    output_tree: str = "DecayTree",
    charge_branch: str = "charge",
    root_include_weights: bool = True,
    root_include_momenta: bool = True,
) -> tuple[PhaseSpaceSample, PhaseSpaceSample]:
    """Generate a CP toy and optionally save both charges in one ROOT TTree.

    The ROOT convention is ``charge=+1`` for the B+ sample and ``charge=-1``
    for the B- sample. The in-memory return value remains ``(plus, minus)``.
    """

    plus_toy, minus_toy = _generate_cp_toy(
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
        method=method,
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
    "ToyBackground",
    "generate_cp_toy",
    "generate_signal_toy",
    "generate_toy",
]
