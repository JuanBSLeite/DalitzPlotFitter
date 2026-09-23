"""D0 -> KS pi+ pi- time-dependent fit and cache benchmark.

Reduced three-resonance model, NOT a reproduction of Belle's full Table I.
Independent NumPy evaluation of Belle Eqs. (1–2) generates an Asimov sample,
or a discrete-quadrature toy with --toy. No experimental input files needed.
"""

import argparse
import json
import time
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from dalitzplotfitter import (
    AmplitudeComponent,
    DecayChannel,
    DecayModel,
    Minimizer,
    NeutralMesonMixing,
    Parameter,
    PreparedAmplitudeCache,
    RealImag,
    Resonance,
    TimeDependentBackgroundCategory,
    TimeDependentDalitzNLL,
    TimeDependentMixtureNLL,
)


@dataclass(frozen=True)
class Reflected:
    """Abar(s12,s13)=A(s13,s12), for ordering (KS,pi+,pi-)."""

    function: object

    def __call__(self, data, parameters=None):
        return self.function(
            {"s12": data["s13"], "s13": data["s12"], "s23": data["s23"]}, parameters
        )


def coefficient(magnitude, phase):
    c = magnitude * np.exp(1j * np.deg2rad(phase))
    return RealImag(c.real, c.imag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=40)
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--backgrounds", action="store_true",
                        help="include two backgrounds and float their fractions")
    args = parser.parse_args()
    model = DecayModel(
        DecayChannel("D0", ("K(S)0", "pi+", "pi-")),
        [
            Resonance(
                "Kstar_minus",
                (0, 2),
                coefficient(1.590, 131.8),
                mass=0.89368,
                width=0.04749,
                spin=1,
            ),
            Resonance(
                "Kstar_plus",
                (0, 1),
                coefficient(0.139, -42.1),
                mass=0.89368,
                width=0.04749,
                spin=1,
            ),
            Resonance("rho", (1, 2), RealImag(1, 0), mass=0.775, width=0.149, spin=1),
        ],
        normalize_components=False,
        normalization_method="square-dalitz",
        normalization_resolution=args.resolution,
    )
    components = tuple(model.amplitude_model.components)
    components += tuple(
        AmplitudeComponent("bar_" + c.name, Reflected(c.function), c.coefficient, False)
        for c in components
    )
    sample = model.normalization_sample
    start = time.perf_counter()
    base = PreparedAmplitudeCache.prepare(
        components,
        data=sample.as_dict(),
        normalization_data=sample.as_dict(),
        normalization_weights=sample.weights,
        normalize_components=False,
    )
    base.normalization_matrix({}).block_until_ready()
    prepare_seconds = time.perf_counter() - start
    groups = jnp.stack((jnp.arange(6) < 3, jnp.arange(6) >= 3), axis=1)
    amplitudes, _ = base.coherent_groups({}, groups)
    a, b = np.asarray(amplitudes).T
    nodes, weights = np.polynomial.legendre.leggauss(60)
    times, time_weights = 2 * (nodes + 1), 2 * weights  # [0,4] ps
    truth = {"mix.x": 0.0056, "mix.y": 0.0030}
    tau = 0.4103
    u = times[:, None] / tau

    def independent_rate(a, b):
        cross = b * a.conj()
        return (
            np.exp(-u)
            / 2
            * (
                (abs(a) ** 2 + abs(b) ** 2) * np.cosh(truth["mix.y"] * u)
                + (abs(a) ** 2 - abs(b) ** 2) * np.cos(truth["mix.x"] * u)
                + 2 * cross.real * np.sinh(truth["mix.y"] * u)
                - 2 * cross.imag * np.sin(truth["mix.x"] * u)
            )
        )

    integration = (
        time_weights[:, None] * np.asarray(sample.weights)[None, :] / sample.size
    )
    probabilities = []
    for aa, bb in ((a, b), (b, a)):
        probability = independent_rate(aa, bb) * integration
        probabilities.append(probability / probability.sum() / 2)
    probabilities = np.stack(probabilities)
    background_pdfs = []
    if args.backgrounds:
        truth.update({"f_sig": 0.8, "f_comb": 0.65})
        for shape, rate in ((np.ones(sample.size), 1.4), (np.asarray(sample.s12), 0.5)):
            dalitz = shape / np.mean(np.asarray(sample.weights) * shape)
            temporal = rate * np.exp(-rate * times) / -np.expm1(-rate * 4)
            background_pdfs.append(temporal[:, None] * dalitz[None, :])
        background_probability = (
            (
                truth["f_comb"] * background_pdfs[0]
                + (1 - truth["f_comb"]) * background_pdfs[1]
            )
            * integration
            / 2
        )
        probabilities = (
            truth["f_sig"] * probabilities
            + (1 - truth["f_sig"]) * background_probability[None, :, :]
        )
    probabilities = probabilities.ravel()
    if args.toy:
        choices = np.random.default_rng(20260923).choice(
            probabilities.size, size=args.events, p=probabilities
        )
        event_weights = jnp.ones(args.events)
    else:
        choices = np.arange(probabilities.size)
        event_weights = jnp.asarray(args.events * probabilities)
    tag_index, time_index, dp_index = np.unravel_index(
        choices, (2, len(times), sample.size)
    )
    # Reuse the already prepared fixed component basis and normalization matrix.
    cache = replace(base, data_components=base.data_components[dp_index])
    parameters = (
        Parameter("mix.x", 0.02, bounds=(-0.2, 0.2), step=0.001),
        Parameter("mix.y", -0.01, bounds=(-0.2, 0.2), step=0.001),
    )
    nll = TimeDependentDalitzNLL(
        cache,
        3,
        times[time_index],
        1 - 2 * tag_index,
        NeutralMesonMixing(parameters[0], parameters[1], tau),
        time_range=(0, 4),
    )
    cache.check_parameters(parameters)
    density = nll.densities
    if args.backgrounds:
        f_sig = Parameter("f_sig", 0.7, bounds=(0.0, 1.0))
        f_comb = Parameter("f_comb", 0.5, bounds=(0.0, 1.0))
        parameters += (f_sig, f_comb)
        mixture = TimeDependentMixtureNLL(
            nll.densities,
            backgrounds=(
                TimeDependentBackgroundCategory(
                    "comb",
                    background_pdfs[0][time_index, dp_index],
                    fraction=f_comb,
                ),
                TimeDependentBackgroundCategory(
                    "partial",
                    background_pdfs[1][time_index, dp_index],
                ),
            ),
            signal_fraction=f_sig,
            tags=nll.tags,
            signal_validity=nll._physical_parameters,
        )
        density = mixture.density

    def objective(values):
        return -jnp.sum(event_weights * jnp.log(density(values)))

    vg = jax.jit(jax.value_and_grad(objective))
    start = time.perf_counter()
    jax.block_until_ready(vg(truth))
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(args.repeats):
        jax.block_until_ready(vg(truth))
    evaluation_seconds = (time.perf_counter() - start) / args.repeats
    result = Minimizer(objective, parameters).fit(strategy=1, hesse=True)
    report = {
        "sample": "quadrature toy" if args.toy else "Asimov",
        "model": "reduced 3-resonance demonstration; not full Belle model",
        "events": args.events,
        "backgrounds": args.backgrounds,
        "evaluation_points": len(choices),
        "prepare_seconds": prepare_seconds,
        "compile_seconds": compile_seconds,
        "value_gradient_seconds": evaluation_seconds,
        "valid": bool(result.valid),
        "truth": truth,
        "fit": {p.name: float(result.values[p.name]) for p in parameters},
        "errors": {p.name: float(result.errors[p.name]) for p in parameters},
    }
    print(json.dumps(report, indent=2))
    if not result.valid:
        raise RuntimeError("Fit did not converge")
    if not args.toy:
        np.testing.assert_allclose(
            list(report["fit"].values()), list(truth.values()), atol=2e-6, rtol=0
        )


if __name__ == "__main__":
    main()
