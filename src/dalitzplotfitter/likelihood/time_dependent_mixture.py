"""Mixtures of tag-conditional, normalized Dalitz-time densities."""

from dataclasses import dataclass

import jax.numpy as jnp

from .mixture import MultiBackgroundNLL, _resolve


@dataclass(frozen=True)
class TimeDependentBackgroundCategory:
    """Prepared joint Dalitz-time background, conditional on tag and sigma_t.

    ``density(parameters)`` returns one normalized density per fitted event.
    Arrays are also accepted for fixed, precomputed densities. The caller
    supplies the normalization, acceptance, veto and time response. For
    projections, optional ``dalitz_density(data, parameters)`` and
    ``time_density(data, parameters)`` return normalized marginal densities;
    data contains ``tag`` and optionally ``sigma_t`` plus invariants or ``t``.

    Fraction/yield semantics match MultiBackgroundNLL. In extended mode,
    ``tag_fraction`` is P(tag=+1) within this category; None uses the signal's
    tag fraction at the low level. The session instead fills None from the
    observed tag fraction. Declare parameters of callable PDFs in
    ``parameters`` so the session can collect them automatically.
    """

    name: str
    density: object
    fraction: object = None
    yield_: object = None
    tag_fraction: object = None
    dalitz_density: object = None
    time_density: object = None
    parameters: tuple = ()

    def __post_init__(self):
        if not self.name:
            raise ValueError("background category name must be non-empty")
        if self.fraction is not None and self.yield_ is not None:
            raise ValueError("a background cannot define both fraction and yield")
        if not callable(self.density):
            values = jnp.asarray(self.density)
            if values.ndim != 1 or bool(jnp.any(~jnp.isfinite(values) | (values < 0))):
                raise ValueError(
                    "background density must be finite nonnegative 1D values"
                )
        for callback in (self.dalitz_density, self.time_density):
            if callback is not None and not callable(callback):
                raise TypeError("background marginal densities must be callable")


@dataclass(frozen=True)
class TimeDependentMixtureNLL(MultiBackgroundNLL):
    """MultiBackgroundNLL with dynamic backgrounds and optional joint tag yields.

    In non-extended mode, mixture fractions are conditional on tag (the same
    fractions for both tags). Each component integrates to one over Dalitz
    and selected observed time for every fixed tag/sigma_t.

    In extended mode, yields count BOTH tags: each component is multiplied by
    its P(tag), making the intensity normalized over (Dalitz, time, tag).
    ``signal_validity(parameters)`` guards the signal's physical domain before
    evaluating the mixture, so a background cannot mask invalid mixing values.
    """

    tags: object = None
    signal_tag_fraction: object = 0.5
    signal_validity: object = None

    def __post_init__(self):
        if self.tags is None:
            raise ValueError("time-dependent mixture requires tags")
        tags = jnp.asarray(self.tags)
        if (
            tags.ndim != 1
            or tags.size == 0
            or not bool(jnp.all((tags == -1) | (tags == 1)))
        ):
            raise ValueError("tags must be a nonempty array of +1/-1")
        super().__post_init__()

    def _physical_parameters(self, parameters):
        valid = super()._physical_parameters(parameters)
        if self.signal_validity is not None:
            valid &= self.signal_validity(parameters)
        if self.extended:
            for fraction in (self.signal_tag_fraction,) + tuple(
                getattr(c, "tag_fraction", None) for c in self.backgrounds
            ):
                if fraction is None:
                    continue
                value = jnp.asarray(_resolve(fraction, parameters))
                if value.ndim != 0:
                    raise ValueError("tag fractions must be scalar")
                valid &= jnp.isfinite(value) & (value >= 0) & (value <= 1)
        return valid

    def _tag_probability(self, fraction, parameters):
        if fraction is None:
            fraction = self.signal_tag_fraction
        plus = _resolve(fraction, parameters)
        return jnp.where(jnp.asarray(self.tags) == 1, plus, 1 - plus)

    def density(self, parameters):
        signal = jnp.asarray(self.signal_density(parameters))
        if signal.shape != jnp.shape(self.tags):
            raise ValueError("signal density must match tags")
        if self.extended:
            total = (
                _resolve(self.signal_yield, parameters)
                * signal
                * self._tag_probability(self.signal_tag_fraction, parameters)
            )
            weights = [_resolve(c.yield_, parameters) for c in self.backgrounds]
        else:
            fraction = (
                1.0
                if not self.backgrounds
                else _resolve(self.signal_fraction, parameters)
            )
            total = fraction * signal
            weights = (1 - fraction) * self.background_weights(parameters)
        valid = jnp.all(jnp.isfinite(signal) & (signal >= 0))
        for category, weight in zip(self.backgrounds, weights, strict=True):
            pdf = category.density
            density = jnp.asarray(pdf(parameters) if callable(pdf) else pdf)
            if density.shape != signal.shape:
                raise ValueError(
                    "all background densities must match the signal data shape"
                )
            valid &= jnp.all(jnp.isfinite(density) & (density >= 0))
            if self.extended:
                density = density * self._tag_probability(
                    getattr(category, "tag_fraction", None), parameters
                )
            total = total + weight * density
        return jnp.where(valid, total, jnp.nan)


__all__ = ["TimeDependentBackgroundCategory", "TimeDependentMixtureNLL"]
