"""Pure-JAX weighted Monte Carlo generation for three-body phase space."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import secrets

import jax
import jax.numpy as jnp
from jax import Array

from .sample import PhaseSpaceSample
from .vectors import invariant_mass_squared


def _kallen(x, y, z):
    return x**2 + y**2 + z**2 - 2.0 * x * y - 2.0 * x * z - 2.0 * y * z


def _unit_vectors(key: Array, size: int, dtype) -> Array:
    key_cos, key_phi = jax.random.split(key)
    cos_theta = jax.random.uniform(key_cos, (size,), dtype=dtype)
    cos_theta = 2.0 * cos_theta - 1.0
    phi = 2.0 * jnp.pi * jax.random.uniform(key_phi, (size,), dtype=dtype)
    sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta**2, 0.0))
    return jnp.stack(
        (
            sin_theta * jnp.cos(phi),
            sin_theta * jnp.sin(phi),
            cos_theta,
        ),
        axis=-1,
    )


def _boost_from_rest(energy: Array, spatial: Array, beta: Array) -> Array:
    """Boost a four-vector from a rest frame to a frame moving with ``beta``."""

    beta2 = jnp.sum(beta * beta, axis=-1)
    gamma = 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta2, jnp.finfo(beta.dtype).tiny))
    beta_dot_p = jnp.sum(beta * spatial, axis=-1)
    safe_beta2 = jnp.where(beta2 > 0.0, beta2, 1.0)
    spatial_factor = (
        (gamma - 1.0) * beta_dot_p / safe_beta2 + gamma * energy
    )
    boosted_spatial = spatial + spatial_factor[..., None] * beta
    boosted_energy = gamma * (energy + beta_dot_p)
    boosted_spatial = jnp.where(
        (beta2 > 0.0)[..., None], boosted_spatial, spatial
    )
    boosted_energy = jnp.where(beta2 > 0.0, boosted_energy, energy)
    return jnp.concatenate((boosted_energy[..., None], boosted_spatial), axis=-1)


@partial(jax.jit, static_argnames=("size",))
def _generate_three_body(
    key: Array,
    mother_mass: Array,
    masses: Array,
    *,
    size: int,
) -> tuple[Array, Array, Array, Array]:
    """Generate weighted ``P -> 1 2 3`` events in the parent rest frame.

    The generator factorizes three-body phase space as

    ``dPhi3 = ds12/(2*pi) * dPhi2(P -> R12 3) * dPhi2(R12 -> 1 2)``.

    ``s12`` and both two-body solid angles are sampled uniformly. The returned
    importance weight is the exact Lorentz-invariant phase-space measure divided
    by that proposal density. Its overall normalization is therefore physical,
    not merely relative.
    """

    dtype = mother_mass.dtype
    m1, m2, m3 = masses
    s_min = (m1 + m2) ** 2
    s_max = (mother_mass - m3) ** 2
    delta_s = s_max - s_min

    key_s, key_parent_angle, key_decay_angle = jax.random.split(key, 3)
    u = jax.random.uniform(key_s, (size,), dtype=dtype)
    # Avoid the exact two-body thresholds, where a zero Jacobian is expected but
    # unnecessary for Monte Carlo sampling. No event weight or invalid value is
    # masked after generation.
    eps = jnp.finfo(dtype).eps
    u = jnp.clip(u, eps, 1.0 - eps)
    s12 = s_min + delta_s * u
    resonance_mass = jnp.sqrt(s12)

    lambda_parent = jnp.maximum(
        _kallen(mother_mass**2, s12, m3**2),
        0.0,
    )
    p_parent = jnp.sqrt(lambda_parent) / (2.0 * mother_mass)
    resonance_energy = (
        mother_mass**2 + s12 - m3**2
    ) / (2.0 * mother_mass)
    bachelor_energy = (
        mother_mass**2 + m3**2 - s12
    ) / (2.0 * mother_mass)

    parent_direction = _unit_vectors(key_parent_angle, size, dtype)
    resonance_spatial = p_parent[:, None] * parent_direction
    p3 = jnp.concatenate(
        (bachelor_energy[:, None], -resonance_spatial),
        axis=-1,
    )

    lambda_resonance = jnp.maximum(
        _kallen(s12, m1**2, m2**2),
        0.0,
    )
    q = jnp.sqrt(lambda_resonance) / (2.0 * resonance_mass)
    daughter1_energy = (
        s12 + m1**2 - m2**2
    ) / (2.0 * resonance_mass)
    daughter2_energy = (
        s12 + m2**2 - m1**2
    ) / (2.0 * resonance_mass)

    decay_direction = _unit_vectors(key_decay_angle, size, dtype)
    daughter1_spatial_rest = q[:, None] * decay_direction
    daughter2_spatial_rest = -daughter1_spatial_rest

    beta = resonance_spatial / resonance_energy[:, None]
    p1 = _boost_from_rest(daughter1_energy, daughter1_spatial_rest, beta)
    p2 = _boost_from_rest(daughter2_energy, daughter2_spatial_rest, beta)

    # With uniform s12 and two independent isotropic solid angles, the Monte
    # Carlo weight is
    #   Delta(s12) * p_parent * q / (32*pi^3*M*sqrt(s12)).
    # Its sample average estimates the physical three-body phase-space volume.
    weights = (
        delta_s
        * p_parent
        * q
        / (32.0 * jnp.pi**3 * mother_mass * resonance_mass)
    )
    return p1, p2, p3, weights


@dataclass(frozen=True)
class PhaseSpaceMC:
    """Generate weighted three-body phase-space events using JAX only.

    All random sampling, kinematics, boosts, invariant masses and weights remain
    on the active JAX device. A CUDA-enabled JAX installation therefore performs
    generation directly on the GPU without TensorFlow or host-framework copies.
    """

    mother_mass: float
    masses: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.masses) != 3:
            raise ValueError("PhaseSpaceMC requires exactly three final-state masses")
        if self.mother_mass <= 0.0:
            raise ValueError("Mother mass must be positive")
        if min(self.masses) < 0.0:
            raise ValueError("Final-state masses must be non-negative")
        if self.mother_mass <= sum(self.masses):
            raise ValueError("Mother mass must be above the three-body threshold")

    def generate(
        self,
        size: int,
        *,
        seed: int | None = None,
        key: Array | None = None,
        include_momenta: bool = True,
    ) -> PhaseSpaceSample:
        """Return weighted phase-space events in ``(E, px, py, pz)`` order.

        Pass either an integer ``seed`` or a JAX PRNG ``key``. Supplying a key is
        convenient for fully functional JAX workflows; ``seed`` is retained as a
        compact high-level API. If neither is given, system entropy is used only
        to create the JAX key and all numerical work remains in JAX.
        """

        if size <= 0:
            raise ValueError("size must be positive")
        if seed is not None and key is not None:
            raise ValueError("Pass either seed or key, not both")
        if key is None:
            if seed is None:
                seed = secrets.randbits(32)
            key = jax.random.key(int(seed))

        mother_mass = jnp.asarray(self.mother_mass)
        masses = jnp.asarray(self.masses, dtype=mother_mass.dtype)
        p1, p2, p3, weights = _generate_three_body(
            key,
            mother_mass,
            masses,
            size=size,
        )

        sample = PhaseSpaceSample(
            s12=invariant_mass_squared(p1 + p2),
            s13=invariant_mass_squared(p1 + p3),
            s23=invariant_mass_squared(p2 + p3),
            weights=weights,
            p1=p1,
            p2=p2,
            p3=p3,
        )
        return sample if include_momenta else sample.without_momenta()
