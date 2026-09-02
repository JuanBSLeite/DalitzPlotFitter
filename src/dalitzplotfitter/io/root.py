"""ROOT-file input/output helpers based on uproot.

No PyROOT dependency is required. Trees are converted to/from JAX arrays and
ROOT TH2 histograms can be converted directly into ordinary- or Square-Dalitz
histogram efficiency/background models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import jax.numpy as jnp
from jax import Array
import numpy as np
import uproot

from dalitzplotfitter.background import HistogramBackground
from dalitzplotfitter.efficiency import HistogramEfficiency
from dalitzplotfitter.kinematics import PhaseSpaceSample
from dalitzplotfitter.square_histograms import (
    SquareDalitzHistogramBackground,
    SquareDalitzHistogramEfficiency,
)

PathLike = str | Path


def _open_object(file_path: PathLike, object_path: str):
    if not object_path:
        raise ValueError("ROOT object path must be non-empty")
    root_file = uproot.open(file_path)
    try:
        return root_file[object_path]
    except KeyError as exc:
        available = list(root_file.keys())
        raise KeyError(
            f"ROOT object {object_path!r} was not found in {str(file_path)!r}; "
            f"available top-level keys include {available[:20]}"
        ) from exc


def _sample_tree_arrays(
    sample: PhaseSpaceSample,
    *,
    include_weights: bool = True,
    include_momenta: bool = True,
) -> dict[str, np.ndarray]:
    """Convert a ``PhaseSpaceSample`` to flat uproot-writable branches."""

    arrays: dict[str, np.ndarray] = {
        "s12": np.asarray(sample.s12),
        "s13": np.asarray(sample.s13),
        "s23": np.asarray(sample.s23),
    }
    if include_weights:
        arrays["weight"] = np.asarray(sample.weights)

    momenta = {"p1": sample.p1, "p2": sample.p2, "p3": sample.p3}
    have_momenta = [momentum is not None for momentum in momenta.values()]
    if any(have_momenta) and not all(have_momenta):
        raise ValueError("p1, p2 and p3 must all be present or all be absent")
    if include_momenta and all(have_momenta):
        labels = ("E", "PX", "PY", "PZ")
        for name, momentum in momenta.items():
            values = np.asarray(momentum)
            if values.shape != (sample.size, 4):
                raise ValueError(
                    f"{name} must have shape ({sample.size}, 4), got {values.shape}"
                )
            for index, component in enumerate(labels):
                arrays[f"{name}_{component}"] = values[:, index]

    if any(values.shape[0] != sample.size for values in arrays.values()):
        raise ValueError("phase-space sample contains arrays with inconsistent lengths")
    return arrays


def write_phase_space_samples(
    file_path: PathLike,
    samples: Mapping[str, PhaseSpaceSample],
    *,
    include_weights: bool = True,
    include_momenta: bool = True,
) -> Path:
    """Write one or more ``PhaseSpaceSample`` objects to ROOT TTrees with uproot.

    ``samples`` maps tree names to samples. The file is recreated atomically by
    uproot, so an existing file at the same path is replaced.
    """

    path = Path(file_path)
    if not samples:
        raise ValueError("at least one ROOT tree/sample must be supplied")
    if any(not str(tree) for tree in samples):
        raise ValueError("ROOT tree names must be non-empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as root_file:
        for tree, sample in samples.items():
            root_file[str(tree)] = _sample_tree_arrays(
                sample,
                include_weights=include_weights,
                include_momenta=include_momenta,
            )
    return path


def write_phase_space_sample(
    file_path: PathLike,
    sample: PhaseSpaceSample,
    *,
    tree: str = "DecayTree",
    include_weights: bool = True,
    include_momenta: bool = True,
) -> Path:
    """Write one ``PhaseSpaceSample`` to a ROOT TTree with uproot."""

    return write_phase_space_samples(
        file_path,
        {tree: sample},
        include_weights=include_weights,
        include_momenta=include_momenta,
    )


def read_root_tree(
    file_path: PathLike,
    tree: str,
    branches: Sequence[str] | Mapping[str, str],
    *,
    cut: str | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> dict[str, Array]:
    if isinstance(branches, Mapping):
        rename = dict(branches)
        expressions = list(rename.values())
    else:
        expressions = list(branches)
        rename = {name: name for name in expressions}
    if not expressions:
        raise ValueError("at least one ROOT branch must be requested")

    tree_obj = _open_object(file_path, tree)
    arrays = tree_obj.arrays(
        expressions,
        cut=cut,
        entry_start=entry_start,
        entry_stop=entry_stop,
        library="np",
        how=dict,
    )
    result: dict[str, Array] = {}
    for output_name, branch_name in rename.items():
        values = np.asarray(arrays[branch_name])
        if values.dtype == object:
            raise ValueError(
                f"branch {branch_name!r} is jagged/object-valued; amplitude-fit inputs "
                "must be flat scalar or fixed-size numeric branches"
            )
        result[output_name] = jnp.asarray(values)
    return result


def read_phase_space_sample(
    file_path: PathLike,
    tree: str,
    *,
    s12: str = "s12",
    s13: str = "s13",
    s23: str = "s23",
    weight: str | None = None,
    p1: Sequence[str] | None = None,
    p2: Sequence[str] | None = None,
    p3: Sequence[str] | None = None,
    cut: str | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> PhaseSpaceSample:
    branch_map: dict[str, str] = {"s12": s12, "s13": s13, "s23": s23}
    if weight is not None:
        branch_map["weight"] = weight

    momentum_specs = {"p1": p1, "p2": p2, "p3": p3}
    supplied = [spec is not None for spec in momentum_specs.values()]
    if any(supplied) and not all(supplied):
        raise ValueError(
            "p1, p2 and p3 four-momentum branch specifications must be supplied together"
        )
    for label, spec in momentum_specs.items():
        if spec is None:
            continue
        if len(spec) != 4:
            raise ValueError(
                f"{label} must contain exactly four branches ordered as (E, px, py, pz)"
            )
        for index, name in enumerate(spec):
            branch_map[f"{label}_{index}"] = name

    arrays = read_root_tree(
        file_path,
        tree,
        branch_map,
        cut=cut,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    size = int(arrays["s12"].shape[0])
    if any(int(arr.shape[0]) != size for arr in arrays.values()):
        raise ValueError("ROOT input branches have inconsistent lengths")

    def momentum(label: str) -> Array | None:
        if momentum_specs[label] is None:
            return None
        return jnp.stack([arrays[f"{label}_{i}"] for i in range(4)], axis=1)

    weights = arrays.get("weight", jnp.ones((size,), dtype=jnp.float64))
    return PhaseSpaceSample(
        s12=arrays["s12"], s13=arrays["s13"], s23=arrays["s23"], weights=weights,
        p1=momentum("p1"), p2=momentum("p2"), p3=momentum("p3"),
    )


def read_root_histogram2d(file_path: PathLike, histogram: str) -> tuple[Array, Array, Array]:
    obj = _open_object(file_path, histogram)
    try:
        values, x_edges, y_edges = obj.to_numpy(flow=False)
    except (AttributeError, ValueError) as exc:
        raise TypeError(
            f"ROOT object {histogram!r} is not a compatible two-dimensional histogram"
        ) from exc
    values = np.asarray(values)
    if values.ndim != 2:
        raise TypeError(f"ROOT object {histogram!r} is not two-dimensional")
    return jnp.asarray(values), jnp.asarray(x_edges), jnp.asarray(y_edges)


def histogram_efficiency_from_root(
    file_path: PathLike,
    histogram: str,
    *,
    x_variable: str = "s12",
    y_variable: str = "s13",
) -> HistogramEfficiency:
    values, x_edges, y_edges = read_root_histogram2d(file_path, histogram)
    return HistogramEfficiency(
        x_edges=x_edges, y_edges=y_edges, values=values,
        x_variable=x_variable, y_variable=y_variable,
    )


def histogram_background_from_root(
    file_path: PathLike,
    histogram: str,
    *,
    x_variable: str = "s12",
    y_variable: str = "s13",
) -> HistogramBackground:
    values, x_edges, y_edges = read_root_histogram2d(file_path, histogram)
    return HistogramBackground(
        x_edges=x_edges, y_edges=y_edges, values=values,
        x_variable=x_variable, y_variable=y_variable,
    )


def square_dalitz_efficiency_from_root(
    file_path: PathLike,
    histogram: str,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
    pair: tuple[int, int] = (0, 1),
) -> SquareDalitzHistogramEfficiency:
    """Construct an efficiency map from a ROOT TH2 whose axes are ``(m', theta')``."""
    values, mp_edges, tp_edges = read_root_histogram2d(file_path, histogram)
    return SquareDalitzHistogramEfficiency(
        mprime_edges=mp_edges,
        thetaprime_edges=tp_edges,
        values=values,
        mother_mass=mother_mass,
        masses=masses,
        pair=pair,
    )


def square_dalitz_background_from_root(
    file_path: PathLike,
    histogram: str,
    *,
    mother_mass: float,
    masses: tuple[float, float, float],
    pair: tuple[int, int] = (0, 1),
) -> SquareDalitzHistogramBackground:
    """Construct a background map from a ROOT TH2 whose axes are ``(m', theta')``."""
    values, mp_edges, tp_edges = read_root_histogram2d(file_path, histogram)
    return SquareDalitzHistogramBackground(
        mprime_edges=mp_edges,
        thetaprime_edges=tp_edges,
        values=values,
        mother_mass=mother_mass,
        masses=masses,
        pair=pair,
    )


__all__ = [
    "histogram_background_from_root",
    "histogram_efficiency_from_root",
    "read_phase_space_sample",
    "read_root_histogram2d",
    "read_root_tree",
    "square_dalitz_background_from_root",
    "square_dalitz_efficiency_from_root",
    "write_phase_space_sample",
    "write_phase_space_samples",
]
