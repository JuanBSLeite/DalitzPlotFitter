"""Covariant N-body events and recursive, weighted Lorentz phase space.

The measure includes (2*pi)^4 multiplying the four-dimensional delta function.
Particles are labelled: no identical-particle factorial is included.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace
from functools import partial
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .phase_space_mc import _boost_from_rest, _unit_vectors
from .vectors import invariant_mass_squared


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NBodySample:
    """Events with ``momenta.shape == (events, particles, 4)``, ordered E,x,y,z.

    Four-vectors are covariant; ``mass_squared`` and the scalar products in
    ``invariants`` are Lorentz invariant. For four particles ``orientation``
    retains the pseudoscalar sign lost by a representation using only masses.
    Integration always means ``mean(weights * f)``. Unit-weight signal events
    are data, not a normalization sample for their own signal density.
    """

    momenta: Array
    weights: Array

    def __post_init__(self):
        if self.momenta.ndim != 3 or self.momenta.shape[-1] != 4:
            raise ValueError("momenta must have shape (events, particles, 4)")
        if self.nbody < 2 or self.weights.shape != (self.size,):
            raise ValueError("need at least two particles and one weight per event")

    @property
    def size(self):
        return self.momenta.shape[0]

    @property
    def nbody(self):
        return self.momenta.shape[1]

    @property
    def nbytes(self):
        return self.momenta.nbytes + self.weights.nbytes

    def tree_flatten(self):
        return (self.momenta, self.weights), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return cls(*children)

    def mass_squared(self, *indices):
        """Invariant mass squared of a nonempty subset (zero-based indices)."""
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("indices must be nonempty and distinct")
        if any(i < 0 or i >= self.nbody for i in indices):
            raise ValueError("particle index out of range")
        return invariant_mass_squared(jnp.sum(self.momenta[:, indices, :], axis=1))

    def invariants(self):
        """Minkowski Gram matrix; for four bodies also the signed determinant."""
        p = self.momenta
        gram = p[..., 0, None] * p[:, None, :, 0]
        gram = gram - jnp.einsum("nik,njk->nij", p[..., 1:], p[..., 1:])
        result = {"gram": gram}
        if self.nbody == 4:
            result["orientation"] = jnp.linalg.det(p)
        return result

    def as_dict(self):
        """Array mapping for existing amplitude/cache/likelihood interfaces."""
        result = {"momenta": self.momenta}
        # Human-readable invariant names follow the established s12 convention.
        if self.nbody <= 9:
            for count in (2, 3):
                for subset in combinations(range(self.nbody), count):
                    name = "s" + "".join(str(i + 1) for i in subset)
                    result[name] = self.mass_squared(*subset)
        if self.nbody == 4:
            result["orientation"] = jnp.linalg.det(self.momenta)
        return result

    def take(self, indices):
        indices = jnp.atleast_1d(jnp.asarray(indices))
        return replace(
            self, momenta=self.momenta[indices], weights=self.weights[indices]
        )

    def observable(self, name):
        """Mass invariants and default (12)(34) helicity angles for projections."""
        if name.startswith("s") and name[1:].isdigit():
            return self.mass_squared(*(int(i) - 1 for i in name[1:]))
        if self.nbody == 4:
            from .four_body import pair_coordinates

            coordinates = pair_coordinates(self.momenta)
            if name in coordinates:
                return coordinates[name]
        raise KeyError(f"unknown observable {name!r}")

    def validate_integration(self):
        if not self.size or not np.all(np.isfinite(self.momenta)):
            raise ValueError("integration momenta must be nonempty and finite")
        w = np.asarray(self.weights)
        if not np.all(np.isfinite(w) & (w >= 0)) or not np.any(w > 0):
            raise ValueError(
                "integration weights need finite nonnegative positive support"
            )

    def validate_physical(self, parent_mass, daughter_masses, *, atol=1e-8):
        """Host-side mass-shell/conservation validation in any inertial frame."""
        p = np.asarray(self.momenta)
        masses = np.asarray(daughter_masses)
        if masses.shape != (self.nbody,):
            raise ValueError("one daughter mass is required per particle")
        if not self.size or not np.all(np.isfinite(p)) or np.any(p[..., 0] <= 0):
            raise ValueError("physical momenta must be finite with positive energies")
        if not np.allclose(invariant_mass_squared(p), masses**2, atol=atol, rtol=atol):
            raise ValueError("daughter momenta are off shell")
        total = np.sum(p, axis=1)
        if not np.allclose(
            invariant_mass_squared(total), parent_mass**2, atol=atol, rtol=atol
        ):
            raise ValueError("total invariant mass disagrees with parent mass")

    def with_importance_weights(self, proposal_density):
        q = jnp.asarray(proposal_density)
        if q.shape != (self.size,) or not bool(jnp.all(jnp.isfinite(q) & (q > 0))):
            raise ValueError("proposal_density must be finite, positive, shape (N,)")
        result = replace(self, weights=1 / q)
        result.validate_integration()
        return result

    def select_for_integration(self, mask):
        self.validate_integration()
        mask = jnp.asarray(mask)
        if mask.shape != (self.size,) or mask.dtype != jnp.bool_:
            raise ValueError("integration selection must be boolean, shape (N,)")
        selected = self.take(jnp.flatnonzero(mask))
        result = replace(selected, weights=selected.weights * selected.size / self.size)
        result.validate_integration()
        return result


@partial(jax.jit, static_argnames=("size",))
def _generate_nbody(key, parent_mass, masses, *, size):
    """Top-down invariant-mass recursion; each two-body decay is isotropic."""
    dtype = masses.dtype
    current = jnp.zeros((size, 4), dtype=dtype).at[:, 0].set(parent_mass)
    current_mass = jnp.full((size,), parent_mass, dtype=dtype)
    weights = jnp.ones((size,), dtype=dtype)
    daughters = [None] * masses.shape[0]
    for k in range(masses.shape[0] - 1, 0, -1):
        key, mass_key, angle_key = jax.random.split(key, 3)
        if k == 1:
            cluster_mass = jnp.full((size,), masses[0])
        else:
            low = jnp.sum(masses[:k]) ** 2
            high = (current_mass - masses[k]) ** 2
            eps = jnp.finfo(dtype).eps
            u = jax.random.uniform(
                mass_key, (size,), dtype=dtype, minval=eps, maxval=1 - eps
            )
            cluster_mass = jnp.sqrt(low + (high - low) * u)
            weights = weights * (high - low) / (2 * jnp.pi)
        # Factorized Kallen form is more stable close to threshold.
        q2 = (
            (current_mass - cluster_mass - masses[k])
            * (current_mass + cluster_mass + masses[k])
            * (current_mass - cluster_mass + masses[k])
            * (current_mass + cluster_mass - masses[k])
        )
        q = jnp.sqrt(jnp.maximum(q2, 0)) / (2 * current_mass)
        weights = weights * q / (4 * jnp.pi * current_mass)
        spatial = q[:, None] * _unit_vectors(angle_key, size, dtype)
        beta = current[:, 1:] / current[:, :1]
        daughters[k] = _boost_from_rest(jnp.sqrt(masses[k] ** 2 + q**2), -spatial, beta)
        current = _boost_from_rest(jnp.sqrt(cluster_mass**2 + q**2), spatial, beta)
        current_mass = cluster_mass
    daughters[0] = current
    return NBodySample(jnp.stack(daughters, axis=1), weights)


@dataclass(frozen=True)
class NBodyPhaseSpaceMC:
    """Physical weighted N-body proposal using dPhi_n=dPhi_2 dPhi_(n-1) ds/2pi.

    The sum of daughter masses must be strictly below the parent mass. Weighted
    output is not uniform phase space until unweighted with its weights.
    """

    mother_mass: float
    masses: tuple[float, ...]

    def __post_init__(self):
        m = np.asarray(self.masses)
        if (
            m.ndim != 1
            or len(m) < 2
            or not np.all(np.isfinite(m))
            or np.any(m < 0)
            or not np.isfinite(self.mother_mass)
            or self.mother_mass <= np.sum(m)
        ):
            raise ValueError("need finite nonnegative masses with M > sum(m_i)")

    def generate(self, size, *, seed=None, key=None):
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("size must be a positive integer")
        if seed is not None and key is not None:
            raise ValueError("supply seed or key, not both")
        if key is None:
            key = jax.random.PRNGKey(secrets.randbits(32) if seed is None else seed)
        return _generate_nbody(
            key,
            jnp.asarray(float(self.mother_mass)),
            jnp.asarray(self.masses, dtype=float),
            size=size,
        )
