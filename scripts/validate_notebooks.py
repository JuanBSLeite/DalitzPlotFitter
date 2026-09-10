"""Validate notebook syntax and optionally run reduced smoke copies in isolation.

Usage: python scripts/validate_notebooks.py --smoke --output /tmp/notebook-audit
Smoke results check API compatibility, not convergence or production precision.
Original notebook code, outputs, and running kernels are never executed in place.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class SmokeLimits(ast.NodeTransformer):
    """Reduce literal workload sizes in a temporary executable AST only."""

    def visit_Constant(self, node):
        if type(node.value) is int and 1000 <= node.value < 1_000_000_000:
            return ast.copy_location(ast.Constant(512), node)
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if any(n in {"N_FITS", "N_TOYS"} for n in names):
            node.value = ast.Constant(2)
        if any("NORMALIZATION_RESOLUTION" in n for n in names):
            node.value = ast.Constant(24)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        limits = {"normalization_resolution": 24, "normalization_order_m13": 24,
                  "normalization_order_m23": 24, "normalization_bin_width": .1,
                  "normalization_binning_factor": 2., "inverse_resolution": 32,
                  "projection_size": 512, "ncall": 512}
        for kw in node.keywords:
            if kw.arg in limits:
                kw.value = ast.Constant(limits[kw.arg])
        return node


def cells(path):
    return json.loads(path.read_text())["cells"]


def worker(path, mode):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed[:2])
    import matplotlib.pyplot as plt
    namespace = {"__name__": "__main__"}
    try:
        from IPython.display import display
        namespace["display"] = display
    except ImportError:
        namespace["display"] = print
    if mode == "smoke":
        # Defaults absent from notebook source must also be bounded.
        from dalitzplotfitter import DecayModel
        init = DecayModel.__init__
        def small_init(self, *args, **kwargs):
            kwargs.update(normalization_resolution=24,
                          normalization_order_m13=24, normalization_order_m23=24,
                          normalization_binning_factor=2.)
            return init(self, *args, **kwargs)
        DecayModel.__init__ = small_init
        from dalitzplotfitter.fit import Minimizer
        fit = Minimizer.fit
        def small_fit(self, *args, **kwargs):
            kwargs["ncall"] = 512
            return fit(self, *args, **kwargs)
        Minimizer.fit = small_fit
    for index, cell in enumerate(cells(path)):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        print(f"CELL {index + 1}", flush=True)
        tree = ast.parse(source)
        if mode == "smoke":
            tree = ast.fix_missing_locations(SmokeLimits().visit(tree))
        exec(compile(tree, f"{path}:cell-{index + 1}", "exec"), namespace)
        plt.close("all")
    print("NOTEBOOK_COMPLETE", flush=True)


def audit(path, output, execute, timeout):
    relative = str(path.relative_to(ROOT))
    result = {"notebook": relative, "syntax": "passed", "execution": "not_run"}
    try:
        for i, cell in enumerate(cells(path)):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
    except (SyntaxError, ValueError) as exc:
        result.update(syntax="failed", error=f"cell {i+1}: {exc}")
        return result
    if not execute:
        return result
    # Real-data notebooks cannot be tested by substituting synthetic files.
    missing = []
    for cell in cells(path):
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value.startswith("/home/") and value.endswith(".root") and not Path(value).exists():
                    missing.append(value)
    if missing:
        result.update(execution="blocked_external_data", missing=sorted(set(missing)))
        return result
    log = output / (relative.replace("/", "__") + ".log")
    with tempfile.TemporaryDirectory(prefix="dpf-notebook-") as temp:
        work = Path(temp)
        (work / "src").symlink_to(ROOT / "src", target_is_directory=True)
        (work / "pyproject.toml").symlink_to(ROOT / "pyproject.toml")
        env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true",
                   MPLBACKEND="Agg", MPLCONFIGDIR=str(work / "mpl"))
        with log.open("w") as stream:
            try:
                run = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--worker", str(path), "--mode", execute], cwd=work, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
                result["execution"] = "passed" if run.returncode == 0 else "failed"
                result["returncode"] = run.returncode
            except subprocess.TimeoutExpired:
                result["execution"] = "timeout"
    result["log"] = str(log)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/dpf-notebook-audit"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--match", default="")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.mode)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted([*ROOT.glob("notebooks/**/*.ipynb"), *ROOT.glob("genfit/*.ipynb")])
    paths = [p for p in paths if args.match in str(p.relative_to(ROOT))]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(audit, p, args.output, "smoke" if args.smoke else None,
                               args.timeout) for p in paths]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output / "results.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
