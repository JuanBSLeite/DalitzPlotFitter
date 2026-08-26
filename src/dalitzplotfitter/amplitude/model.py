"""Pure callable wrappers around compiled TensorWaves models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from jax import Array

ParameterMapping = Mapping[str, Array | float | complex]


@dataclass(frozen=True)
class CompiledModel:
    """Pure parameter-explicit view of a TensorWaves compiled function.

    TensorWaves exposes mutable parameter updates for interactive workflows. For a
    fitter we instead want ``f(data, parameters)`` so JAX can trace, differentiate,
    and JIT-compile the whole likelihood without mutating hidden state.
    """

    function: object

    @property
    def parameters(self) -> dict[str, object]:
        return self.function.parameters

    @property
    def argument_order(self) -> tuple[str, ...]:
        return self.function.argument_order

    def __call__(self, data: Mapping[str, Array], parameters: ParameterMapping | None = None):
        values = dict(self.parameters)
        if parameters is not None:
            unknown = set(parameters) - set(values)
            if unknown:
                raise ValueError(f"Unknown model parameters: {sorted(unknown)}")
            values.update(parameters)

        merged = {**data, **values}
        missing = [name for name in self.argument_order if name not in merged]
        if missing:
            raise KeyError(f"Missing model arguments: {missing}")
        args = [merged[name] for name in self.argument_order]
        return self.function.function(*args)
