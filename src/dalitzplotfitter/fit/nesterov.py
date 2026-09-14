"""Projected accelerated gradient prefit for smooth bounded objectives."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class NesterovResult:
    """Minuit-compatible result returned by a Nesterov-only fit."""

    values: Mapping[str, float]
    fval: float
    valid: bool
    status: str
    converged: bool
    history: tuple[Mapping[str, float], ...]
    nfcn: int
    errors: Mapping[str, float]
    covariance: object = None
    optimizer: str = "nesterov"

    @property
    def fmin(self):
        return SimpleNamespace(edm=float("nan"))


def minimize(objective: Callable, parameters: Sequence, *,
             start_values: Mapping[str, float] | None = None,
             max_iter: int = 1000, gtol: float = 1e-4,
             verbose: int = 0) -> NesterovResult:
    """Run monotone projected Nesterov in parameter-scaled coordinates."""
    if max_iter < 1 or gtol <= 0:
        raise ValueError("max_iter must be positive and gtol must be positive")
    supplied = {} if start_values is None else dict(start_values)
    free = tuple(p for p in parameters if not p.fixed)
    fixed = {p.name: float(p.value) for p in parameters if p.fixed}
    origin = np.asarray([supplied.get(p.name, p.value) for p in free], dtype=float)
    scales = np.asarray([p.step if p.step is not None else max(abs(float(origin[i])), 1.0)
                         for i, p in enumerate(free)], dtype=float)
    bounds = [p.bounds or (None, None) for p in free]
    lower = np.asarray([(-np.inf if b[0] is None else b[0] - origin[i]) / scales[i]
                        for i, b in enumerate(bounds)])
    upper = np.asarray([(np.inf if b[1] is None else b[1] - origin[i]) / scales[i]
                        for i, b in enumerate(bounds)])

    def scaled_nll(z):
        physical = jnp.asarray(origin) + jnp.asarray(scales) * z
        values = dict(fixed)
        values.update({p.name: physical[i] for i, p in enumerate(free)})
        return objective(values)

    value_and_grad = jax.jit(jax.value_and_grad(scaled_nll))
    x = np.zeros(len(free), dtype=float)
    evaluations = 0

    def evaluate(z):
        nonlocal evaluations
        f, g = jax.device_get(value_and_grad(jnp.asarray(z)))
        evaluations += 1
        return float(f), np.asarray(g, dtype=float)

    def residual(z, g):
        return float(np.max(np.abs(z - np.clip(z - g, lower, upper))))

    f, g = evaluate(x)
    y, t, step = x.copy(), 1.0, 1.0
    history = [{"iteration": 0, "nll": f, "projected_gradient": residual(x, g)}]
    status = "max_iter"
    for iteration in range(1, max_iter + 1):
        if residual(x, g) <= gtol:
            status = "converged"
            break
        accepted = False
        for attempt in range(2):
            yy = x if attempt else y
            fy, gy = (f, g) if np.array_equal(yy, x) else evaluate(yy)
            trial = step * 1.1
            for _ in range(60):
                candidate = np.clip(yy - trial * gy, lower, upper)
                delta = candidate - yy
                fc, gc = evaluate(candidate)
                majorant = fy + gy @ delta + delta @ delta / (2.0 * trial)
                if (np.isfinite(fc) and np.all(np.isfinite(gc))
                        and fc <= majorant + 1e-12 * max(1.0, abs(fy))
                        and fc <= f):
                    accepted = True
                    break
                trial *= 0.5
            if accepted:
                break
        if not accepted:
            status = "line_search_failed"
            break
        old_x, x, f, g, step = x, candidate, fc, gc, trial
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = np.clip(x + (t - 1.0) / t_next * (x - old_x), lower, upper)
        t = t_next
        history.append({"iteration": iteration, "nll": f,
                        "projected_gradient": residual(x, g)})
        report_every = 50 if verbose == 2 else 10 if verbose >= 3 else None
        if report_every is not None and iteration % report_every == 0:
            print(f"[Nesterov] {iteration}: NLL={f:.9f}, "
                  f"projected gradient={residual(x, g):.4g}", flush=True)
    if residual(x, g) <= gtol:
        status = "converged"
    physical = origin + scales * x
    values = {p.name: float(physical[i]) for i, p in enumerate(free)}
    values.update(fixed)
    return NesterovResult(values=values, fval=f, valid=status == "converged",
                          status=status, converged=status == "converged",
                          history=tuple(history), nfcn=evaluations,
                          errors={p.name: float("nan") for p in parameters if not p.fixed})
