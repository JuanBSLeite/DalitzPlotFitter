"""ROOT-file input helpers based on uproot.

No PyROOT dependency is required.  Trees are converted to JAX arrays and ROOT
TH2 histograms can be converted directly into the package histogram efficiency
and background models.
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


def read_root_tree(
    file_path: PathLike,
    tree: str,
    branches: Sequence[str] | Mapping[str, str],
    *,
    cut: str | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> dict[str, Array]:
    """Read selected TTree branches as JAX arrays.

    ``branches`` may be a sequence, preserving ROOT branch names, or a mapping
    ``{output_name: root_branch_name}`` for convenient renaming.  ``cut`` uses
    uproot's expression filtering.
    """

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
    """Load a three-body fit sample from a ROOT TTree.

    Four-momentum specifications, when supplied, must each contain four branch
    names ordered as ``(E, px, py, pz)``.  If ``weight`` is omitted the event
    weights are set to one, which is appropriate for ordinary unweighted data.
    """

    branch_map: dict[str, str] = {"s12": s12, "s13": s13, "s23": s23}
    if weight is not None:
        branch_map["weight"] = weight

    momentum_specs = {"p1": p1, "p2": p2, "p3": p3}
    supplied = [spec is not None for spec in momentum_specs.values()]
    if any(supplied) and not all(supplied):
        raise ValueError("p1, p2 and p3 four-momentum branch specifications must be supplied together")
    for label, spec in momentum_specs.items():
        if spec is None:
            continue
        if len(spec) != 4:
            raise ValueError(f"{label} must contain exactly four branches ordered as (E, px, py, pz)")
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
        s12=arrays["s12"],
        s13=arrays["s13"],
        s23=arrays["s23"],
        weights=weights,
        p1=momentum("p1"),
        p2=momentum("p2"),
        p3=momentum("p3"),
    )


def read_root_histogram2d(
    file_path: PathLike,
    histogram: str,
) -> tuple[Array, Array, Array]:
    """Read a ROOT TH2-like object as ``(values, x_edges, y_edges)``."""

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
    """Construct :class:`HistogramEfficiency` directly from a ROOT TH2."""

    values, x_edges, y_edges = read_root_histogram2d(file_path, histogram)
    return HistogramEfficiency(
        x_edges=x_edges,
        y_edges=y_edges,
        values=values,
        x_variable=x_variable,
        y_variable=y_variable,
    )


def histogram_background_from_root(
    file_path: PathLike,
    histogram: str,
    *,
    x_variable: str = "s12",
    y_variable: str = "s13",
) -> HistogramBackground:
    """Construct :class:`HistogramBackground` directly from a ROOT TH2."""

    values, x_edges, y_edges = read_root_histogram2d(file_path, histogram)
    return HistogramBackground(
        x_edges=x_edges,
        y_edges=y_edges,
        values=values,
        x_variable=x_variable,
        y_variable=y_variable,
    )


__all__ = [
    "histogram_background_from_root",
    "histogram_efficiency_from_root",
    "read_phase_space_sample",
    "read_root_histogram2d",
    "read_root_tree",
]
