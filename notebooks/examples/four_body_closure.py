"""Independent continuous toy + MC-normalized four-body closure.

Run: python notebooks/examples/four_body_closure.py --events 6000 --normalization 200000
Uses two scalar chains and the existing fixed-width Pole convention. The
rejection bound below is analytical for this particular model, not a generic
envelope estimate. No toy candidate pool is reused for normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from jaxpwa import (
    AmplitudeComponent,
    CascadeChain,
    FitSession,
    FourBodyDecayModel,
    Isobar,
    NBodyDecayChannel,
    NBodyPhaseSpaceMC,
    NBodySample,
    PairChain,
    Parameter,
    RealImag,
)

TRUTH = {"cascade.x": 0.55, "cascade.y": 0.35}
CHANNEL = NBodyDecayChannel(2.0, (0.1, 0.2, 0.3, 0.4))


def make_model(normalization):
    pair = PairChain(Isobar(0.65, 0.3), Isobar(0.95, 0.35))
    cascade = CascadeChain(Isobar(1.45, 0.35), Isobar(0.95, 0.35))
    return FourBodyDecayModel(
        CHANNEL,
        [
            AmplitudeComponent("pair", pair, RealImag(1.0, 0.0)),
            AmplitudeComponent(
                "cascade",
                cascade,
                RealImag(
                    Parameter.coefficient(
                        "cascade.x", 0.25, bounds=(-2.0, 2.0), step=0.05
                    ),
                    Parameter.coefficient(
                        "cascade.y", 0.1, bounds=(-2.0, 2.0), step=0.05
                    ),
                ),
            ),
        ],
        normalization_sample=normalization,
    )


def generate_toy(model, size, *, seed=714, batch_size=20_000):
    """Exact accept-reject draws conditional on the model's fixed component scales."""
    generator = NBodyPhaseSpaceMC(CHANNEL.parent_mass, CHANNEL.daughter_masses)
    template = model.prepare_cache(model.normalization_sample.take(jnp.array([0])))
    scales = template.component_scales
    functions = tuple(c.function for c in model.components)
    coefficients = jnp.array([1.0, complex(TRUTH["cascade.x"], TRUTH["cascade.y"])])

    @jax.jit
    def intensity(p):
        amplitude = sum(
            scales[i] * coefficients[i] * f({"momenta": p})
            for i, f in enumerate(functions)
        )
        return jnp.abs(amplitude) ** 2

    # At each isotropic two-body step Phi2 = q/(4pi M) <= 1/(8pi).
    # Each sampled ds interval is bounded by the maximal available cluster mass.
    masses, mother = CHANNEL.daughter_masses, CHANNEL.parent_mass
    ps_bound = 1 / (8 * np.pi) ** 3
    for k in (3, 2):
        upper = (mother - sum(masses[k:])) ** 2
        lower = sum(masses[:k]) ** 2
        ps_bound *= (upper - lower) / (2 * np.pi)
    # For Pole(m): |Pole| <= 2/Gamma; every spin-zero angular/radial factor is 1.
    amplitude_bound = float(scales[0]) * 4 / (0.3 * 0.35) + float(scales[1]) * abs(
        complex(coefficients[1])
    ) * 4 / (0.35 * 0.35)
    envelope = ps_bound * amplitude_bound**2 * (1 + 1e-12)
    key = jax.random.PRNGKey(seed)
    accepted, count, generated = [], 0, 0
    for _ in range(10_000):
        key, ps_key, accept_key = jax.random.split(key, 3)
        sample = generator.generate(batch_size, key=ps_key)
        target = sample.weights * intensity(sample.momenta)
        if not bool(jnp.all(jnp.isfinite(target) & (target <= envelope))):
            raise RuntimeError("toy envelope violated; never clip an invalid envelope")
        mask = jax.random.uniform(accept_key, (batch_size,)) * envelope < target
        events = sample.momenta[mask]
        accepted.append(events)
        count += len(events)
        generated += batch_size
        if count >= size:
            return NBodySample(
                jnp.concatenate(accepted)[:size], jnp.ones(size)
            ), generated
    raise RuntimeError("toy generation exhausted its batch limit")


def run_closure(*, events=6000, normalization=200_000, output=None):
    start = perf_counter()
    norm = NBodyPhaseSpaceMC(CHANNEL.parent_mass, CHANNEL.daughter_masses).generate(
        normalization, seed=812
    )
    model = make_model(norm)
    data, generated = generate_toy(model, events)
    session = FitSession(model, data)
    result = session.fit(hesse=True, strategy=1)
    values = session.result_values(result)
    pulls = {
        name: (values[name] - truth) / result.errors[name]
        for name, truth in TRUTH.items()
    }
    summary = dict(
        valid=bool(result.valid),
        events=events,
        normalization=normalization,
        proposals=generated,
        truth=TRUTH,
        fitted=values,
        errors={name: result.errors[name] for name in TRUTH},
        pulls=pulls,
        elapsed_seconds=perf_counter() - start,
    )
    print(json.dumps(summary, indent=2))
    if not result.valid or any(abs(pull) > 5 for pull in pulls.values()):
        raise AssertionError("four-body closure failed")
    if output is not None:
        import matplotlib.pyplot as plt

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "closure.json").write_text(json.dumps(summary, indent=2) + "\n")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for ax, variable in zip(axes, ("s12", "s234", "phi"), strict=True):
            session.plot_projection(
                result, variable, bins=35, projection_sample=norm, ax=ax
            )
            ax.set_xlabel(variable)
        fig.savefig(output / "closure.png", dpi=140)
        plt.close(fig)
    return session, result, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=6000)
    parser.add_argument("--normalization", type=int, default=200_000)
    parser.add_argument("--output", default="output/four_body_closure")
    args = parser.parse_args()
    run_closure(
        events=args.events, normalization=args.normalization, output=args.output
    )
