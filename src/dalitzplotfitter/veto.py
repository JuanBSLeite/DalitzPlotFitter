"""Dalitz-plot veto maps.

Vetoes are represented by an acceptance mask equal to one in accepted regions
and zero in vetoed regions.  This lets the same object be applied consistently
to data selection, toy generation and PDF normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.kinematics import PhaseSpaceSample


class VetoMap:
    """Protocol-like base class for Dalitz acceptance masks."""

    def __call__(self, data: dict[str, Array]) -> Array:
        raise NotImplementedError

    def accept(self, data: dict[str, Array]) -> Array:
        return jnp.asarray(self(data), dtype=bool)

    def apply(self, sample: PhaseSpaceSample) -> PhaseSpaceSample:
        indices = jnp.nonzero(self.accept(sample.as_dict()), size=None)[0]
        return sample.take(indices)


@dataclass(frozen=True)
class FunctionalVeto(VetoMap):
    """Wrap a callable returning ``True`` for accepted Dalitz points."""

    function: Callable[[dict[str, Array]], Array]

    def __call__(self, data: dict[str, Array]) -> Array:
        mask = jnp.asarray(self.function(data), dtype=bool)
        expected = jnp.asarray(next(iter(data.values()))).shape[0]
        if mask.shape != (expected,):
            raise ValueError(f"veto function must return shape ({expected},), got {mask.shape}")
        return mask


@dataclass(frozen=True)
class MassWindowVeto(VetoMap):
    """Veto one invariant-mass window, Laura++ ``addMassVeto`` style."""

    pair: tuple[int, int]
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        i, j = sorted(self.pair)
        if (i, j) not in ((0, 1), (0, 2), (1, 2)):
            raise ValueError("pair must contain two distinct indices from 0, 1, 2")
        if self.minimum < 0.0 or self.maximum <= self.minimum:
            raise ValueError("mass veto requires 0 <= minimum < maximum")
        object.__setattr__(self, "pair", (i, j))

    @property
    def variable(self) -> str:
        return {(0, 1): "s12", (0, 2): "s13", (1, 2): "s23"}[self.pair]

    def __call__(self, data: dict[str, Array]) -> Array:
        mass = jnp.sqrt(jnp.maximum(jnp.asarray(data[self.variable]), 0.0))
        vetoed = (mass >= self.minimum) & (mass <= self.maximum)
        return ~vetoed


@dataclass(frozen=True)
class CompositeVeto(VetoMap):
    """Logical AND of any number of veto maps."""

    vetoes: tuple[VetoMap, ...]

    def __init__(self, *vetoes: VetoMap):
        object.__setattr__(self, "vetoes", tuple(vetoes))

    def __call__(self, data: dict[str, Array]) -> Array:
        size = jnp.asarray(next(iter(data.values()))).shape[0]
        mask = jnp.ones((size,), dtype=bool)
        for veto in self.vetoes:
            mask = mask & veto.accept(data)
        return mask


@dataclass(frozen=True)
class VetoedDensity:
    """Apply a veto map to any Dalitz density/shape callable."""

    density: Callable[[dict[str, Array]], Array]
    veto: VetoMap

    def __call__(self, data: dict[str, Array]) -> Array:
        values = jnp.asarray(self.density(data))
        mask = jnp.asarray(self.veto(data), dtype=values.dtype)
        return values * mask


def vetoed_signal_pdf(model, veto: VetoMap, *, normalization_sample=None, efficiency=None):
    """Build a veto-aware ``SignalPDF`` directly from a ``DecayModel``."""

    from dalitzplotfitter.integration import GridIntegrator
    from dalitzplotfitter.pdf import SignalPDF

    sample = model.normalization_sample if normalization_sample is None else normalization_sample

    def intensity(data, parameters):
        return model.intensity(data, parameters)

    kwargs = {"veto": veto}
    if efficiency is not None:
        kwargs["efficiency"] = efficiency
    return SignalPDF(
        intensity=intensity,
        integrator=GridIntegrator(sample),
        **kwargs,
    )


__all__ = [
    "CompositeVeto",
    "FunctionalVeto",
    "MassWindowVeto",
    "VetoMap",
    "VetoedDensity",
    "vetoed_signal_pdf",
]
