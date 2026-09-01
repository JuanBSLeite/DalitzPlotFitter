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

    The true signal density is ``rho_true = epsilon(phi_true)|A|^2``.  The
    correctly reconstructed part stays at the same Dalitz point, while the SCF
    part migrates through :class:`SquareDalitzSCFMap`.

    If ``veto`` is supplied it is applied in reconstructed coordinates.  Thus
    an SCF event that migrates into a vetoed reconstructed bin is rejected.  In
    that case the total normalization is recomputed as accepted CR probability
    plus accepted migrated-SCF probability rather than assuming full migration
    conservation.
    """

    intensity: ParametricIntensity
    integrator: GridIntegrator
    scf_map: SquareDalitzSCFMap
    efficiency: object = UnityEfficiency()
    veto: object | None = None
    floor: float = 1e-300

    def _acceptance(self, data: dict[str, Array]) -> Array:
        if self.veto is None:
            first = jnp.asarray(next(iter(data.values())))
            return jnp.ones(first.shape[0], dtype=first.dtype)
        return jnp.asarray(self.veto(data), dtype=jnp.float64)

    def _true_bin_density(self, parameters: Parameters) -> Array:
        true_data = self.scf_map.true_bin_data()
        return self.efficiency(true_data) * self.intensity(true_data, parameters)

    def normalization(self, parameters: Parameters) -> Array:
        if self.veto is None:
            return self.integrator.integrate(
                lambda data: self.efficiency(data) * self.intensity(data, parameters)
            )

        def cr_integrand(data):
            base = self.efficiency(data) * self.intensity(data, parameters)
            f_scf = self.scf_map.scf_fraction_at(data["s12"], data["s13"], data["s23"])
            return self._acceptance(data) * (1.0 - f_scf) * base

        cr_norm = self.integrator.integrate(cr_integrand)

        true_density = self._true_bin_density(parameters)
        scf_density = self.scf_map.smeared_bin_density(true_density)
        reco_data = self.scf_map.true_bin_data()
        reco_acceptance = self._acceptance(reco_data)
        scf_norm = jnp.sum(
            reco_acceptance * scf_density * self.scf_map.phase_space_areas()
        )
        return cr_norm + scf_norm

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
        return self._acceptance(data) * (correctly_reconstructed + scf)

    def __call__(self, data: dict[str, Array], parameters: Parameters) -> Array:
        numerator = self.numerator(data, parameters)
        normalized = numerator / self.normalization(parameters)
        return jnp.where(numerator > 0.0, jnp.clip(normalized, min=self.floor), 0.0)

    def logpdf(self, data: dict[str, Array], parameters: Parameters) -> Array:
        numerator = self.numerator(data, parameters)
        return jnp.where(
            numerator > 0.0,
            jnp.log(jnp.clip(numerator, min=self.floor)) - jnp.log(self.normalization(parameters)),
            -jnp.inf,
        )


__all__ = ["SCFSignalPDF"]
