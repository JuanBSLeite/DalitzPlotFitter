"""Reproduce histogram ablations for analysis notebook 22 (local inputs required).

Example:
    python benchmarks/diagnose_laura_comparison.py --variant corrected --fit

The original and jacobian variants reconstruct the pre-audit binwise maps;
all variants start at the printed final Laura parameters. HESSE is omitted:
this diagnoses shifts of the minimum, not uncertainties.
"""

# ruff: noqa: E402
# Imports follow repository path setup so this script works without an editable install.

import argparse
import ast
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "notebooks" / "data_analyses"))

import jax.numpy as jnp
import uproot
from laura_comparison_helpers import charge_scaled_background

from dalitzplotfitter import (
    SquareDalitzHistogramBackground,
    SquareDalitzHistogramEfficiency,
    square_dalitz_jacobian,
)


def local_tree(file_path, tree, branches):
    # A local file object avoids the fsspec async reader hang seen with this
    # environment's Python 3.14/uproot combination. Values are unchanged.
    with open(file_path, "rb") as handle, uproot.open(handle) as root:
        arrays = root[tree].arrays(list(branches.values()), library="np")
        return {key: jnp.asarray(arrays[value]) for key, value in branches.items()}


def binwise(histogram, cls):
    return cls(**{field.name: getattr(histogram, field.name) for field in fields(cls)})


def converted_binwise(histogram):
    def shape(data):
        mp, tp = histogram.square_coordinates(data)
        jac = square_dalitz_jacobian(
            mp,
            tp,
            mother_mass=histogram.mother_mass,
            masses=histogram.masses,
            pair=histogram.pair,
        )
        return jnp.where(jac > 0, histogram(data) / jnp.where(jac > 0, jac, 1), 0)

    return shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("original", "jacobian", "corrected"), default="corrected"
    )
    parser.add_argument("--resolution", type=int, default=300)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    notebook = json.loads(
        (
            REPO / "notebooks/data_analyses/22_b2pipipi_cpvfit_laura_comparison.ipynb"
        ).read_text()
    )
    scope = {"REPO_ROOT": REPO}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("result = fit_session.fit"):
            break
        exec(compile(source, f"notebook-cell-{index}", "exec"), scope)
        scope["read_root_tree"] = local_tree
        scope["NORMALIZATION_RESOLUTION"] = args.resolution
    session = scope["fit_session"]
    if args.variant != "corrected":
        session = replace(session, backgrounds=()).with_efficiency(
            binwise(scope["plus_efficiency"], SquareDalitzHistogramEfficiency),
            binwise(scope["minus_efficiency"], SquareDalitzHistogramEfficiency),
        )
        for name, hp, hm, asym, total in (
            (
                "qqbar",
                scope["_qqbar_plus_pdf"],
                scope["_qqbar_minus_pdf"],
                scope["QQBAR_YIELD_ASYMMETRY"],
                scope["N_QQBAR"],
            ),
            (
                "kpipibkg",
                scope["_kpipibkg_pdf"],
                scope["_kpipibkg_pdf"],
                scope["KPIPIBKG_YIELD_ASYMMETRY"],
                0.0,
            ),
        ):
            shapes = []
            for hist, charge, model in (
                (hp, +1, session.plus_model),
                (hm, -1, session.minus_model),
            ):
                hist = binwise(hist, SquareDalitzHistogramBackground)
                if args.variant == "original":
                    area = jnp.outer(
                        jnp.diff(hist.mprime_edges), jnp.diff(hist.thetaprime_edges)
                    )
                    hist = replace(
                        hist,
                        values=(1 - charge * asym)
                        * hist.values
                        / jnp.sum(hist.values * area),
                    )
                    shapes.append(hist)
                else:
                    shapes.append(
                        charge_scaled_background(
                            converted_binwise(hist),
                            model.normalization_sample,
                            scope["D0_VETO"],
                            asym,
                            charge,
                        )
                    )
            session = session.with_background(
                name,
                shapes[0],
                minus_shape=shapes[1],
                yield_=scope["Parameter"]("n_" + name, total, fixed=True),
            )
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "LAURA_FINAL" for t in node.targets
            ):
                laura = ast.literal_eval(node.value)
    start = {p.name: float(p.value) for p in session.parameters}
    start.update({key: value[0] for key, value in laura.items()})
    output = {
        "variant": args.variant,
        "resolution": args.resolution,
        "laura_point_nll": float(session.objective(start)),
        "qqbar_plus_probability": float(
            session.background_categories[0].plus_probability
        ),
    }
    print(json.dumps(output), flush=True)
    if args.fit:
        result = session.fit(start_values=start, strategy=1, hesse=False, verbose=1)
        output.update(values=session.result_values(result), valid=bool(result.valid))
        output["nll"] = float(session.objective(output["values"]))
        output["delta_nll_from_laura_point"] = output["nll"] - output["laura_point_nll"]
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
