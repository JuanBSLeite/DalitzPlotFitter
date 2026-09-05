"""Generic weighted three-body event container."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class PhaseSpaceSample:
    """Three-body events, invariants, four-momenta and Monte Carlo weights."""

    s12: Array
    s13: Array
    s23: Array
    weights: Array
    p1: Array | None = None
    p2: Array | None = None
    p3: Array | None = None

    @property
    def size(self) -> int:
        return int(self.s12.shape[0])

    def as_dict(self) -> dict[str, Array]:
        data = {"s12": self.s12, "s13": self.s13, "s23": self.s23}
        if self.p1 is not None and self.p2 is not None and self.p3 is not None:
            data.update({"p1": self.p1, "p2": self.p2, "p3": self.p3})
        return data

    def momentum_dict(self) -> dict[str, Array]:
        if self.p1 is None or self.p2 is None or self.p3 is None:
            raise ValueError("This sample does not contain four-momenta")
        return {"p1": self.p1, "p2": self.p2, "p3": self.p3}

    @property
    def nbytes(self) -> int:
        """Approximate bytes occupied by the sample's array payloads."""

        arrays = (self.s12, self.s13, self.s23, self.weights, self.p1, self.p2, self.p3)
        return sum(
            int(jnp.asarray(array).size * jnp.asarray(array).dtype.itemsize)
            for array in arrays
            if array is not None
        )

    def without_momenta(self) -> "PhaseSpaceSample":
        """Return a compact view containing only invariants and event weights."""

        if self.p1 is None and self.p2 is None and self.p3 is None:
            return self
        return PhaseSpaceSample(
            s12=self.s12,
            s13=self.s13,
            s23=self.s23,
            weights=self.weights,
        )

    def take(self, indices: Array) -> "PhaseSpaceSample":
        indices = jnp.asarray(indices)
        return PhaseSpaceSample(
            s12=self.s12[indices],
            s13=self.s13[indices],
            s23=self.s23[indices],
            weights=self.weights[indices],
            p1=None if self.p1 is None else self.p1[indices],
            p2=None if self.p2 is None else self.p2[indices],
            p3=None if self.p3 is None else self.p3[indices],
        )
