"""Regression checks for public API usage in shipped notebooks."""

from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK_DIR = Path("notebooks")
TOY_FUNCTIONS = {"generate_toy", "generate_signal_toy", "generate_cp_toy"}


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def test_notebook_toy_sampler_options_match_selected_method():
    """Accept-reject-only knobs must never be passed to inverse-transform."""

    failures: list[str] = []
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            tree = ast.parse(source, filename=f"{path}:cell-{cell_index}")
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or _call_name(node) not in TOY_FUNCTIONS:
                    continue
                keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
                if not ({"pool_size", "batch_size"} & keywords.keys()):
                    continue
                method = _constant_string(keywords.get("method"))
                if method != "accept-reject":
                    failures.append(
                        f"{path}:cell-{cell_index}: pool_size/batch_size require "
                        "method='accept-reject'"
                    )

    assert not failures, "\n".join(failures)


def test_notebooks_do_not_describe_accept_reject_as_default():
    stale_phrases = (
        "accept-reject is the default",
        "accept-reject remains the default",
        "default toy sampler is the laura++-style accept-reject",
    )
    failures: list[str] = []
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "markdown":
                continue
            source = "".join(cell.get("source", [])).lower()
            if any(phrase in source for phrase in stale_phrases):
                failures.append(f"{path}:cell-{cell_index}")

    assert not failures, "stale toy-sampler documentation in: " + ", ".join(failures)
