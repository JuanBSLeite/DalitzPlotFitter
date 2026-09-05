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
def _generate_three_body_invariants(
    key: Array,
    mother_mass: Array,
    masses: Array,
    *,
    size: int,
) -> tuple[Array, Array, Array, Array]:
    """Generate the same weighted phase-space proposal directly in invariants.

    Sampling s12 uniformly and a uniform coordinate across its exact s13
    limits is equivalent to the invariant projection of the full generator.
    The importance weight is unchanged up to the same physical normalization,
    while orientations, boosts, and four-vectors are never constructed.
    """

    dtype = mother_mass.dtype
    m1, m2, m3 = masses
    s_min = (m1 + m2) ** 2
    s_max = (mother_mass - m3) ** 2
    delta_s = s_max - s_min

    key_s, key_v = jax.random.split(key)
    eps = jnp.finfo(dtype).eps
    u = jnp.clip(
        jax.random.uniform(key_s, (size,), dtype=dtype),
        eps,
        1.0 - eps,
    )
    v = jax.random.uniform(key_v, (size,), dtype=dtype)
    s12 = s_min + delta_s * u

    root_s12 = jnp.sqrt(s12)
    e1 = (s12 + m1**2 - m2**2) / (2.0 * root_s12)
    e3 = (mother_mass**2 - s12 - m3**2) / (2.0 * root_s12)
    q = jnp.sqrt(jnp.maximum(_kallen(s12, m1**2, m2**2), 0.0)) / (
        2.0 * root_s12
    )
    p = jnp.sqrt(
        jnp.maximum(_kallen(mother_mass**2, s12, m3**2), 0.0)
    ) / (2.0 * root_s12)

    common = m1**2 + m3**2 + 2.0 * e1 * e3
    spread = 2.0 * q * p
    low = common - spread
    high = common + spread
    width = high - low
    s13 = low + v * width
    constant = mother_mass**2 + m1**2 + m2**2 + m3**2
    s23 = constant - s12 - s13

    # Exact match to the full generator weight:
    # width(s13 | s12) = sqrt(lambda_parent*lambda_resonance) / s12.
    weights = delta_s * width / (128.0 * jnp.pi**3 * mother_mass**2)
    return s12, s13, s23, weights


@partial(
    jax.jit,
    static_argnames=("size", "u_bins", "v_bins"),
)
def _generate_three_body_invariant_cells(
    key: Array,
    mother_mass: Array,
    masses: Array,
    cell_probabilities: Array,
    *,
    size: int,
    u_bins: int,
    v_bins: int,
) -> tuple[Array, Array, Array, Array, Array]:
    """Generate invariant proposals from equal cells in the unit Dalitz square."""

    if u_bins < 1 or v_bins < 1:
        raise ValueError("stratified phase-space bins must be positive")
    n_cells = u_bins * v_bins
    probabilities = jnp.asarray(cell_probabilities)
    if probabilities.shape != (n_cells,):
        raise ValueError(
            "cell_probabilities must contain one probability per stratified cell"
        )

    dtype = mother_mass.dtype
    key_cell, key_u, key_v = jax.random.split(key, 3)
    cell = jax.random.choice(
        key_cell,
        n_cells,
        shape=(size,),
        p=probabilities,
        replace=True,
    )
    iu = cell // v_bins
    iv = cell % v_bins

    ru = jax.random.uniform(key_u, (size,), dtype=dtype)
    rv = jax.random.uniform(key_v, (size,), dtype=dtype)
    u = (iu.astype(dtype) + ru) / float(u_bins)
    v = (iv.astype(dtype) + rv) / float(v_bins)

    m1, m2, m3 = masses
    s_min = (m1 + m2) ** 2
    s_max = (mother_mass - m3) ** 2
    delta_s = s_max - s_min
    eps = jnp.finfo(dtype).eps
    u = jnp.clip(u, eps, 1.0 - eps)
    s12 = s_min + delta_s * u

    root_s12 = jnp.sqrt(s12)
    e1 = (s12 + m1**2 - m2**2) / (2.0 * root_s12)
    e3 = (mother_mass**2 - s12 - m3**2) / (2.0 * root_s12)
    q = jnp.sqrt(jnp.maximum(_kallen(s12, m1**2, m2**2), 0.0)) / (
        2.0 * root_s12
    )
    p = jnp.sqrt(
        jnp.maximum(_kallen(mother_mass**2, s12, m3**2), 0.0)
    ) / (2.0 * root_s12)
    common = m1**2 + m3**2 + 2.0 * e1 * e3
    spread = 2.0 * q * p
    low = common - spread
    width = 2.0 * spread
    s13 = low + v * width
    constant = mother_mass**2 + m1**2 + m2**2 + m3**2
    s23 = constant - s12 - s13
    weights = delta_s * width / (128.0 * jnp.pi**3 * mother_mass**2)
    return s12, s13, s23, weights, cell.astype(jnp.int32)


@partial(jax.jit, static_argnames=("size",))
def _momenta_from_invariants(
    key: Array,
    mother_mass: Array,
    masses: Array,
    s12: Array,
    s13: Array,
    s23: Array,
    *,
    size: int,
) -> tuple[Array, Array, Array]:
    """Reconstruct an isotropic parent-rest-frame orientation from invariants."""

    dtype = mother_mass.dtype
    m1, m2, m3 = masses
    mother2 = mother_mass**2

    e1 = (mother2 + m1**2 - s23) / (2.0 * mother_mass)
    e2 = (mother2 + m2**2 - s13) / (2.0 * mother_mass)
    e3 = (mother2 + m3**2 - s12) / (2.0 * mother_mass)
    p1_mag = jnp.sqrt(jnp.maximum(e1**2 - m1**2, 0.0))
    p2_mag = jnp.sqrt(jnp.maximum(e2**2 - m2**2, 0.0))

    pair_dot = 0.5 * (s12 - m1**2 - m2**2)
    spatial_dot = e1 * e2 - pair_dot
    denominator = p1_mag * p2_mag
    cos12 = jnp.where(denominator > 0.0, spatial_dot / denominator, 1.0)
    cos12 = jnp.clip(cos12, -1.0, 1.0)
    sin12 = jnp.sqrt(jnp.maximum(1.0 - cos12**2, 0.0))

    key_n1, key_alpha = jax.random.split(key)
    n1 = _unit_vectors(key_n1, size, dtype)

    reference_z = jnp.broadcast_to(
        jnp.asarray((0.0, 0.0, 1.0), dtype=dtype), n1.shape
    )
    reference_x = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0), dtype=dtype), n1.shape
    )
    reference = jnp.where(
        (jnp.abs(n1[:, 2]) > 0.9)[:, None], reference_x, reference_z
    )
    e_perp1 = jnp.cross(reference, n1)
    norm = jnp.linalg.norm(e_perp1, axis=1)
    e_perp1 = e_perp1 / norm[:, None]
    e_perp2 = jnp.cross(n1, e_perp1)

    alpha = 2.0 * jnp.pi * jax.random.uniform(key_alpha, (size,), dtype=dtype)
    transverse = (
        jnp.cos(alpha)[:, None] * e_perp1
        + jnp.sin(alpha)[:, None] * e_perp2
    )
    n2 = cos12[:, None] * n1 + sin12[:, None] * transverse

    spatial1 = p1_mag[:, None] * n1
    spatial2 = p2_mag[:, None] * n2
    spatial3 = -(spatial1 + spatial2)
    p1 = jnp.concatenate((e1[:, None], spatial1), axis=1)
    p2 = jnp.concatenate((e2[:, None], spatial2), axis=1)
    p3 = jnp.concatenate((e3[:, None], spatial3), axis=1)
    return p1, p2, p3


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

        if not include_momenta:
            s12, s13, s23, weights = _generate_three_body_invariants(
                key,
                mother_mass,
                masses,
                size=size,
            )
            return PhaseSpaceSample(
                s12=s12,
                s13=s13,
                s23=s23,
                weights=weights,
            )

        p1, p2, p3, weights = _generate_three_body(
            key,
            mother_mass,
            masses,
            size=size,
        )
        return PhaseSpaceSample(
            s12=invariant_mass_squared(p1 + p2),
            s13=invariant_mass_squared(p1 + p3),
            s23=invariant_mass_squared(p2 + p3),
            weights=weights,
            p1=p1,
            p2=p2,
            p3=p3,
        )

    def generate_stratified_invariants(
        self,
        size: int,
        *,
        cell_probabilities: Array,
        grid_shape: tuple[int, int],
        seed: int | None = None,
        key: Array | None = None,
    ) -> tuple[PhaseSpaceSample, Array]:
        """Generate invariant-only proposals from weighted equal Dalitz cells."""

        if size <= 0:
            raise ValueError("size must be positive")
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must contain (u_bins, v_bins)")
        u_bins, v_bins = (int(grid_shape[0]), int(grid_shape[1]))
        if u_bins < 1 or v_bins < 1:
            raise ValueError("grid_shape entries must be positive")
        probabilities = jnp.asarray(cell_probabilities)
        if probabilities.shape != (u_bins * v_bins,):
            raise ValueError(
                "cell_probabilities must match the requested stratified grid"
            )
        if seed is not None and key is not None:
            raise ValueError("Pass either seed or key, not both")
        if key is None:
            if seed is None:
                seed = secrets.randbits(32)
            key = jax.random.key(int(seed))

        mother_mass = jnp.asarray(self.mother_mass)
        masses = jnp.asarray(self.masses, dtype=mother_mass.dtype)
        s12, s13, s23, weights, cells = _generate_three_body_invariant_cells(
            key,
            mother_mass,
            masses,
            probabilities,
            size=size,
            u_bins=u_bins,
            v_bins=v_bins,
        )
        return (
            PhaseSpaceSample(
                s12=s12,
                s13=s13,
                s23=s23,
                weights=weights,
            ),
            cells,
        )

    def attach_momenta(
        self,
        sample: PhaseSpaceSample,
        *,
        seed: int | None = None,
        key: Array | None = None,
    ) -> PhaseSpaceSample:
        """Attach isotropic four-momenta to an invariant-only physical sample."""

        if sample.p1 is not None or sample.p2 is not None or sample.p3 is not None:
            if sample.p1 is None or sample.p2 is None or sample.p3 is None:
                raise ValueError("sample contains an incomplete set of four-momenta")
            return sample
        if seed is not None and key is not None:
            raise ValueError("Pass either seed or key, not both")
        if key is None:
            if seed is None:
                seed = secrets.randbits(32)
            key = jax.random.key(int(seed))

        mother_mass = jnp.asarray(self.mother_mass)
        masses = jnp.asarray(self.masses, dtype=mother_mass.dtype)
        p1, p2, p3 = _momenta_from_invariants(
            key,
            mother_mass,
            masses,
            jnp.asarray(sample.s12),
            jnp.asarray(sample.s13),
            jnp.asarray(sample.s23),
            size=sample.size,
        )
        return PhaseSpaceSample(
            s12=sample.s12,
            s13=sample.s13,
            s23=sample.s23,
            weights=sample.weights,
            p1=p1,
            p2=p2,
            p3=p3,
        )
