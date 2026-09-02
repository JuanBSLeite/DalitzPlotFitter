"""ROOT output helpers specialized for generated pseudo-data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import uproot

from dalitzplotfitter.kinematics import PhaseSpaceSample

from .root import PathLike, _sample_tree_arrays


def write_cp_phase_space_sample(
    file_path: PathLike,
    plus_sample: PhaseSpaceSample,
    minus_sample: PhaseSpaceSample,
    *,
    tree: str = "DecayTree",
    charge_branch: str = "charge",
    include_weights: bool = True,
    include_momenta: bool = True,
) -> Path:
    """Write B+ and B- samples to one TTree with a signed charge branch.

    The output convention is ``charge=+1`` for the plus sample and
    ``charge=-1`` for the minus sample.
    """

    if not tree:
        raise ValueError("ROOT tree name must be non-empty")
    if not charge_branch:
        raise ValueError("charge branch name must be non-empty")

    plus = _sample_tree_arrays(
        plus_sample,
        include_weights=include_weights,
        include_momenta=include_momenta,
    )
    minus = _sample_tree_arrays(
        minus_sample,
        include_weights=include_weights,
        include_momenta=include_momenta,
    )
    if set(plus) != set(minus):
        raise ValueError("B+ and B- samples must expose the same ROOT branches")
    if charge_branch in plus:
        raise ValueError(f"charge branch {charge_branch!r} conflicts with a sample branch")

    arrays = {
        name: np.concatenate((plus[name], minus[name]), axis=0)
        for name in plus
    }
    arrays[charge_branch] = np.concatenate(
        (
            np.ones(plus_sample.size, dtype=np.int8),
            -np.ones(minus_sample.size, dtype=np.int8),
        )
    )

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as root_file:
        root_file.mktree(tree, arrays)
    return path


__all__ = ["write_cp_phase_space_sample"]
