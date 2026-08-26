"""Fit-parameter declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    """Configuration for one scalar fit parameter."""

    name: str
    value: float
    fixed: bool = False
    bounds: tuple[float | None, float | None] | None = None
