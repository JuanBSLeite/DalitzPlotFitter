"""Signal PDF with Laura++-style self-cross-feed migration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
from jax import Array

from dalitzplotfitter.efficiency import UnityEfficiency
from dalitzplotfitter.integration import GridIntegrator
from dalitzplotfitter.resolution import SquareDalitzSCFMap

Parameters = Mapping[str, Array | float]
ParametricIntensity = Callable[[dict[str, Array], Parameters], Array]


@dataclass(frozen=True)
class SCFSignalPDF:
    """Normalized signal PDF including correctly reconstructed and SCF events.

    The true signal density is

    ``rho_true = epsilon(phi_true) |A(phi_true)|^2``.

    The correctly reconstructed term keeps the fraction ``1-f_SCF`` at the
    same Dalitz position.  The SCF term migrates ``f_SCF * rho_true`` through
    :class:`SquareDalitzSCFMap` from true to reconstructed Square-Dalitz bins.

    Because each true-bin migration distribution is normalized to unity,
    ``CR + SCF`` conserves the total signal probability.  Therefore the
    normalization remains ``integral epsilon |A|^2 dPhi``.
    """

    intensity: ParametricIntensity
    integrator: GridIntegrator
    scf_map: SquareDalitzSCFMap
    efficiency: object = UnityEfficiency()
    floor: float = 1e-300

    def normalization(self, parameters: Parameters) -> Array:
        return self.integrator.integrate(
            lambda data: self.efficiency(data) * self.intensity(data, parameters)
        )

    def _true_bin_density(self, parameters: Parameters) -> Array:
        true_data = self.scf_map.true_bin_data()
        return self.efficiency(true_data) * self.intensity(true_data, parameters)

    def numerator(self, data: dict[str, Array], parameters: Parameters) -> Array:
        required = ("s12", "s13", "s23")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"SCFSignalPDF requires invariant keys {required}; missing {missing}")

        base = self.efficiency(data) * self.intensity(data, parameters)
        f_scf = self.scf_map.scf_fraction_at(
            data["s12"], data["s13"], data["s23"]
        )
        correctly_reconstructed = (1.0 - f_scf) * base

        scf = self.scf_map.smeared_density_at(
            self._true_bin_density(parameters),
            data["s12"],
            data["s13"],
            data["s23"],
        )
        return correctly_reconstructed + scf

    def __call__(self, data: dict[str, Array], parameters: Parameters) -> Array:
        return jnp.clip(
            self.numerator(data, parameters) / self.normalization(parameters),
            min=self.floor,
        )

    def logpdf(self, data: dict[str, Array], parameters: Parameters) -> Array:
        return jnp.log(jnp.clip(self.numerator(data, parameters), min=self.floor)) - jnp.log(
            self.normalization(parameters)
        )


__all__ = ["SCFSignalPDF"]
