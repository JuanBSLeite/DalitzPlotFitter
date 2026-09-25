"""Audit saved notebook 27 at its printed (rounded) best-fit parameters.

Requires the notebook's local ROOT maps. Executes setup cells only, never fits,
and leaves the notebook untouched. Run from the repository root on CPU.
"""

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions", nargs="+", type=int, default=[500, 1000, 1500, 2000]
    )
    parser.add_argument("--mass-orders", nargs="*", type=int, default=[24, 48, 96])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = Path("notebooks/data_analyses/27_b2pipipi_cpvfit_qmi_step_cp_bins.ipynb")
    notebook = json.loads(path.read_text())
    scope = {}

    def run(index, stop=None):
        source = "".join(notebook["cells"][index]["source"])
        if stop:
            source = source.split(stop)[0]
        exec(compile(source, f"notebook27-cell-{index}", "exec"), scope)

    for index in (1, 3, 5, 9, 11, 13):
        run(index)
    run(15, "isobar_session =")
    printed = "".join(
        "".join(o.get("text", [])) for o in notebook["cells"][35]["outputs"]
    )
    values = {
        name: float(value)
        for name, value in re.findall(
            r"^([\w.]+)\s+([-+\d.eE]+)\s+[-+\d.eE]+\s*$", printed, re.M
        )
    }
    if "S_QMI.bin_24.x" not in values:
        raise ValueError("Saved parameter table missing or notebook layout changed")
    scope["isobar_values"] = values
    run(22)
    run(26)
    import jax
    import jax.numpy as jnp
    import numpy as np

    from dalitzplotfitter import SquareDalitzGrid
    from dalitzplotfitter.integration.gauss_legendre import _s13_limits_numpy
    from dalitzplotfitter.kinematics import PhaseSpaceSample

    models = (scope["plus_model"], scope["minus_model"])
    efficiencies = (scope["plus_efficiency"], scope["minus_efficiency"])
    linear_efficiencies = tuple(
        replace(e, interpolation="linear") for e in efficiencies
    )
    backgrounds = (scope["_qqbar_plus_pdf"], scope["_qqbar_minus_pdf"])
    veto = scope["D0_VETO"]

    @jax.jit
    def evaluate(data, weights):
        accepted = veto(data)
        strict = jnp.ones_like(accepted)
        for key in ("s13", "s23"):
            strict &= ~((data[key] > 1.74**2) & (data[key] < 1.894**2))
        result = [
            jnp.sum(weights),
            jnp.sum(weights * accepted),
            jnp.sum(accepted != strict),
        ]
        for model, efficiency, linear_efficiency, background in zip(
            models, efficiencies, linear_efficiencies, backgrounds, strict=True
        ):
            components = jnp.stack(
                [
                    c.coefficient.value(values) * c.function(data, values)
                    for c in model.amplitude_model.components
                ],
                axis=1,
            )
            intensity = jnp.abs(components.sum(axis=1)) ** 2
            result.extend(
                [
                    jnp.sum(weights * intensity),
                    jnp.sum(weights * accepted * efficiency(data) * intensity),
                    jnp.sum(weights * accepted * linear_efficiency(data) * intensity),
                    jnp.sum(weights * accepted * background(data)),
                ]
            )
            result.extend(jnp.sum(weights[:, None] * jnp.abs(components) ** 2, axis=0))
        return jnp.stack(result)

    labels = ["area", "accepted_area", "strict_boundary_different_nodes"]
    for charge, model in zip(("plus", "minus"), models, strict=True):
        labels += [
            charge + "_physical",
            charge + "_accepted",
            charge + "_accepted_linear_efficiency",
            charge + "_background",
        ]
        labels += [
            charge + "_component_" + c.name for c in model.amplitude_model.components
        ]
    output = {
        "notebook": str(path),
        "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "efficiency_interpolation": [e.interpolation for e in efficiencies],
        "background_interpolation": [b.interpolation for b in backgrounds],
        "parameters": values,
        "note": "Rounded saved parameters; current saved source and maps; no refit.",
        "grids": [],
    }

    def samples():
        yield (
            "square-midpoint",
            1000,
            SquareDalitzGrid(
                scope["MOTHER_MASS"],
                scope["DAUGHTER_MASSES"],
                resolution=1000,
                pair=(0, 1),
                quadrature="midpoint",
            ).sample(),
        )
        for resolution in args.resolutions:
            yield (
                "square-gauss",
                resolution,
                SquareDalitzGrid(
                    scope["MOTHER_MASS"],
                    scope["DAUGHTER_MASSES"],
                    resolution=resolution,
                    pair=(0, 1),
                    quadrature="gauss-legendre",
                ).sample(),
            )
        # Independent mass-plane quadrature, partitioned at ALL QMI/veto edges
        # and narrow-resonance windows. This is not Laura's exact partition.
        mother, masses = scope["MOTHER_MASS"], scope["DAUGHTER_MASSES"]
        low, high = masses[0] + masses[2], mother - masses[1]
        edges = list(scope["QMI_KNOTS_GEV"])
        for name in ("omega_782", "chi_c0"):
            edges += [
                values[name + ".mass"] + sign * 5 * values[name + ".width"]
                for sign in (-1, 1)
            ]
        edges = np.array(sorted({low, high, *(e for e in edges if low < e < high)}))
        for order in args.mass_orders:
            x, w = np.polynomial.legendre.leggauss(order)
            half = np.diff(edges) / 2
            nodes = ((edges[:-1] + half)[:, None] + half[:, None] * x).ravel()
            weights = (half[:, None] * w).ravel()
            m13, m23 = np.meshgrid(nodes, nodes, indexing="ij")
            raw = 4 * m13 * m23 * weights[:, None] * weights[None, :]
            s13, s23 = m13.ravel() ** 2, m23.ravel() ** 2
            s12 = mother**2 + sum(m * m for m in masses) - s13 - s23
            lower, upper = _s13_limits_numpy(
                np.maximum(s12, (masses[0] + masses[1]) ** 2),
                mother_mass=mother,
                masses=masses,
            )
            valid = (
                (s12 >= (masses[0] + masses[1]) ** 2)
                & (s12 <= (mother - masses[2]) ** 2)
                & (s13 >= lower)
                & (s13 <= upper)
            )
            count = int(valid.sum())
            yield (
                "mass-split-gauss",
                order,
                PhaseSpaceSample(
                    s12=jnp.asarray(s12[valid]),
                    s13=jnp.asarray(s13[valid]),
                    s23=jnp.asarray(s23[valid]),
                    weights=jnp.asarray(raw.ravel()[valid] * count),
                ),
            )

    for method, resolution, sample in samples():
        total = np.zeros(len(labels))
        chunk = 50000
        for start in range(0, sample.size, chunk):
            stop = min(start + chunk, sample.size)
            data = {
                k: jnp.asarray(v[start:stop])
                for k, v in sample.as_dict().items()
                if k in ("s12", "s13", "s23")
            }
            weights = sample.weights[start:stop] / sample.size
            pad = chunk - (stop - start)
            data = {k: jnp.pad(v, (0, pad), mode="edge") for k, v in data.items()}
            weights = jnp.pad(weights, (0, pad))
            total += np.asarray(evaluate(data, weights))
        row = {
            "method": method,
            "resolution_or_cell_order": resolution,
            "points": sample.size,
            **dict(zip(labels, total.tolist(), strict=True)),
        }
        output["grids"].append(row)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
