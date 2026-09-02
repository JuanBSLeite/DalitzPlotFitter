"""Compatibility facade for the public toy-generation API.

Toy generation has exactly two public methods, ``accept-reject`` and
``inverse-transform``. The implementations live in ``toy_accept`` and
``toy_inverse`` and are dispatched by ``toy_api``.
"""

from .toy_api import (
    CPToyBackground,
    PreparedInverseToyGenerator,
    ToyBackground,
    generate_cp_toy,
    generate_signal_toy,
    generate_toy,
    prepare_inverse_toy_generator,
)

__all__ = [
    "CPToyBackground",
    "PreparedInverseToyGenerator",
    "ToyBackground",
    "generate_cp_toy",
    "generate_signal_toy",
    "generate_toy",
    "prepare_inverse_toy_generator",
]
