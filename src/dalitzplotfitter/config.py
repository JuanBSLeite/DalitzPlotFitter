"""Numerical configuration for DalitzPlotFitter."""

from __future__ import annotations

from jax import config as jax_config


def enable_x64(enabled: bool = True) -> None:
    """Enable or disable JAX 64-bit floating-point precision.

    Importing ``dalitzplotfitter`` already calls this with ``enabled=True`` (unless
    the ``JAX_ENABLE_X64`` environment variable was set explicitly before import,
    which is left authoritative). Call this directly only to opt back out, e.g. for
    an explicit, validated float32 experiment.
    """

    jax_config.update("jax_enable_x64", enabled)
