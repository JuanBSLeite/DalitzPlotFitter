"""Signal plus arbitrary background-category likelihoods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.background import BackgroundCategory

Parameters = Mapping[str, Array | float]
SignalDensity = Callable[[Parameters], Array]


def _resolve(value: object, parameters: Parameters):
    resolver = getattr(value, "resolve", None)
    return resolver(parameters) if resolver is not None else value


@dataclass(frozen=True)
class MultiBackgroundNLL:
    """Unbinned signal plus arbitrary named background categories.

    Non-extended convention
    -----------------------
    ``signal_fraction`` is the total signal fraction.  Background-category
    ``fraction`` parameters describe the *relative composition of the total
    background*.  For ``N`` categories the first ``N-1`` categories carry
    fractions and the last category is the remainder.  Thus

    ``p = f_sig S + (1-f_sig) sum_k w_k B_k`` with ``sum_k w_k = 1``.

    Extended convention
    -------------------
    ``signal_yield`` and every category's ``yield_`` are independent expected
    yields:

    ``lambda = N_sig S + sum_k N_k B_k``.
    """

    signal_density: SignalDensity
    backgrounds: tuple[BackgroundCategory, ...] = ()
    signal_fraction: object | None = None
    extended: bool = False
    signal_yield: object | None = None

    def __post_init__(self) -> None:
        names = [category.name for category in self.backgrounds]
        if len(set(names)) != len(names):
            raise ValueError("background category names must be unique")
        if self.extended:
            if self.signal_fraction is not None:
                raise ValueError("signal_fraction is not used in extended mode")
            if self.signal_yield is None:
                raise ValueError("extended mode requires signal_yield")
            if any(category.fraction is not None for category in self.backgrounds):
                raise ValueError("background fractions are not used in extended mode")
            if any(category.yield_ is None for category in self.backgrounds):
                raise ValueError("every extended background category requires yield_")
        else:
            if self.signal_yield is not None:
                raise ValueError("signal_yield requires extended=True")
            if self.backgrounds and self.signal_fraction is None:
                raise ValueError("background mixtures require signal_fraction")
            if any(category.yield_ is not None for category in self.backgrounds):
                raise ValueError("background yields require extended=True")
            if len(self.backgrounds) > 1:
                if any(c.fraction is None for c in self.backgrounds[:-1]):
                    raise ValueError(
                        "all background categories except the last require a relative fraction"
                    )
                if self.backgrounds[-1].fraction is not None:
                    raise ValueError(
                        "the last background category is the remainder and must not define fraction"
                    )
            elif len(self.backgrounds) == 1 and self.backgrounds[0].fraction is not None:
                raise ValueError("a single background category does not need a relative fraction")

    def background_weights(self, parameters: Parameters) -> Array:
        """Return relative background-category weights summing to one."""

        n = len(self.backgrounds)
        if n == 0:
            return jnp.empty((0,), dtype=jnp.float64)
        if n == 1:
            return jnp.ones((1,), dtype=jnp.float64)
        explicit = jnp.asarray(
            [_resolve(category.fraction, parameters) for category in self.backgrounds[:-1]],
            dtype=jnp.float64,
        )
        remainder = 1.0 - jnp.sum(explicit)
        return jnp.concatenate((explicit, jnp.asarray([remainder])))

    def density(self, parameters: Parameters) -> Array:
        signal = jnp.asarray(self.signal_density(parameters))
        if not self.backgrounds:
            if self.extended:
                return jnp.asarray(_resolve(self.signal_yield, parameters)) * signal
            return signal

        background_densities = jnp.stack([category.density for category in self.backgrounds])
        if background_densities.shape[1:] != signal.shape:
            raise ValueError("all background densities must match the signal data shape")

        if self.extended:
            total = jnp.asarray(_resolve(self.signal_yield, parameters)) * signal
            for category, density in zip(self.backgrounds, background_densities):
                total = total + jnp.asarray(_resolve(category.yield_, parameters)) * density
            return total

        f_signal = jnp.asarray(_resolve(self.signal_fraction, parameters))
        weights = self.background_weights(parameters)
        background = jnp.sum(weights[:, None] * background_densities, axis=0)
        return f_signal * signal + (1.0 - f_signal) * background

    def expected_events(self, parameters: Parameters) -> Array:
        if not self.extended:
            raise RuntimeError("expected_events is only defined in extended mode")
        total = jnp.asarray(_resolve(self.signal_yield, parameters))
        for category in self.backgrounds:
            total = total + jnp.asarray(_resolve(category.yield_, parameters))
        return total

    def __call__(self, parameters: Parameters) -> Array:
        density = self.density(parameters)
        tiny = jnp.finfo(density.dtype).tiny
        nll = -jnp.sum(jnp.log(jnp.maximum(density, tiny)))
        if self.extended:
            nll = nll + self.expected_events(parameters)
        return nll


__all__ = ["MultiBackgroundNLL"]
