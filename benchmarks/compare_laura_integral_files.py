"""Compare raw isobar integrals against Laura's initialization .dat files.

Uses initial Laura 3.8 resonance catalog values, the supplied .cc overrides,
bilinear acceptance, and normalize_form_factors=False. Analytic conversion
factors from the default convention are reported, not fitted to the reference.
Run from the repository root with local inputs.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from dalitzplotfitter import (  # noqa: E402
    CompositeVeto,
    DecayChannel,
    DecayModel,
    GounarisSakurai,
    MassWindowVeto,
    PhaseSpaceSample,
    RelativisticBreitWigner,
    Resonance,
    SigmaPole,
    SquareDalitzGrid,
    SquareDalitzHistogramEfficiency,
    ZemachP,
    read_root_histogram2d,
)
from dalitzplotfitter.integration.gauss_legendre import _s13_limits_numpy  # noqa: E402

CATALOG = [
    ("rho0(770)", 0.77526, 0.1478, 1, GounarisSakurai),
    ("omega(782)", 0.78265, 0.00849, 1, RelativisticBreitWigner),
    ("rho0(1450)", 1.465, 0.4, 1, GounarisSakurai),
    ("f_2(1270)", 1.2755, 0.1867, 2, RelativisticBreitWigner),
    ("rho0_3(1690)", 1.686, 0.186, 3, RelativisticBreitWigner),
    ("sigma0", 0.475, 0.55, 0, SigmaPole),
    ("f_0(980)", 0.990, 0.070, 0, RelativisticBreitWigner),
    ("chi_c0", 3.41471, 0.0131, 0, RelativisticBreitWigner),
    ("f_0(1370)", 1.370, 0.350, 0, RelativisticBreitWigner),
]


def read_integrals(path):
    lines = path.read_text().splitlines()
    size = int(lines[0].split()[-1])
    matrices = []
    for line in lines[6:8]:
        pairs = re.findall(r"\(([^,]+),([^\)]+)\)", line)
        if len(pairs) != size * (size + 1) // 2:
            raise ValueError("Unexpected Laura upper-triangle length")
        matrix = np.zeros((size, size), complex)
        indices = np.triu_indices(size)
        matrix[indices] = [complex(float(a), float(b)) for a, b in pairs]
        matrix += np.triu(matrix, 1).conj().T
        matrices.append(matrix)
    np.testing.assert_allclose(
        np.diag(matrices[0]).real, np.fromstring(lines[5], sep=" ")
    )
    np.testing.assert_allclose(
        np.diag(matrices[1]).real, np.fromstring(lines[4], sep=" ")
    )
    return lines[1].split(), matrices[1], matrices[0]


def encode(matrix):
    return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}


def logged_grid(directory, mother, pion):
    """Reproduce the logged partitions and local Laura 3.8 Gauss algorithm.

    Interior endpoints are printed exactly for these two narrow windows;
    replace rounded physical endpoints with the explicitly selected masses.
    Deliberately retains Laura's stale zSq and 1e-6 stopping criterion.
    """
    log = (directory / "GenFit3piCP_fit_415489.log").read_text()
    log = log.split("INFO in LauIsobarDynamics::writeIntegralsFile")[0]
    pattern = (
        r"nm13Points = (\d+), nm23Points = (\d+).*?"
        r"Integrating over m13 = ([\d.e+-]+) to ([\d.e+-]+), "
        r"m23 = ([\d.e+-]+) to ([\d.e+-]+)"
    )
    regions = re.findall(pattern, log, re.S)
    if len(regions) != 25:
        raise ValueError(f"Expected 25 logged rectangles, found {len(regions)}")

    def axis(n, lo, hi):
        n = int(n)
        lo, hi = float(lo), float(hi)
        if abs(lo - 2 * pion) < 1e-5:
            lo = 2 * pion
        if abs(hi - (mother - pion)) < 1e-5:
            hi = mother - pion
        z = np.cos(np.pi * (np.arange(1, (n + 1) // 2 + 1) - 0.25) / (n + 0.5))
        zsq = z * z
        active = np.ones(len(z), bool)
        derivative = np.zeros_like(z)
        for _ in range(100):
            p1, p2 = np.ones_like(z), np.zeros_like(z)
            for j in range(1, n + 1):
                p1, p2 = ((2 * j - 1) * z * p1 - (j - 1) * p2) / j, p1
            pp = n * (z * p1 - p2) / (zsq - 1)
            updated = z - p1 / pp
            derivative[active] = pp[active]
            delta = abs(updated - z)
            z[active] = updated[active]
            active &= delta > 1e-6
            if not active.any():
                break
        else:
            raise ValueError("Laura quadrature did not converge")
        w = 2 / ((1 - zsq) * derivative**2)
        half = (hi - lo) / 2
        nodes, weights = np.zeros(n), np.zeros(n)
        for i in range(len(z)):
            nodes[i], nodes[n - 1 - i] = (
                (lo + hi) / 2 - half * z[i],
                (lo + hi) / 2 + half * z[i],
            )
            weights[i] = weights[n - 1 - i] = half * w[i]
        return nodes, weights

    columns = [[], [], [], []]
    for n, m, lo, hi, low, high in regions:
        x, wx = axis(n, lo, hi)
        y, wy = axis(m, low, high)
        a, b = np.meshgrid(x, y, indexing="ij")
        s13, s23 = a.ravel() ** 2, b.ravel() ** 2
        s12 = mother**2 + 3 * pion**2 - s13 - s23
        lower, upper = _s13_limits_numpy(
            np.maximum(s12, 4 * pion**2), mother_mass=mother, masses=(pion,) * 3
        )
        valid = (
            (s12 >= 4 * pion**2)
            & (s12 <= (mother - pion) ** 2)
            & (s13 >= lower)
            & (s13 <= upper)
        )
        raw = (4 * a * b * wx[:, None] * wy[None, :]).ravel()
        for target, values in zip(columns, (s12, s13, s23, raw), strict=True):
            target.append(values[valid])
    s12, s13, s23, weights = [np.concatenate(c) for c in columns]
    return PhaseSpaceSample(
        s12=jnp.asarray(s12),
        s13=jnp.asarray(s13),
        s23=jnp.asarray(s23),
        weights=jnp.asarray(weights * len(weights)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(
            "/home/juan-leite/Work/data/fit415489_pipipinotav3_21-5--11h32m13s415489"
        ),
    )
    parser.add_argument("--resolutions", nargs="+", type=int, default=[500, 1000, 2000])
    parser.add_argument("--parent-mass", type=float, default=5.27934)
    parser.add_argument("--pion-mass", type=float, default=0.1395704)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    class ReferenceChannel(DecayChannel):
        @property
        def parent_mass(self):
            return args.parent_mass

        @property
        def daughter_masses(self):
            return (args.pion_mass,) * 3

    channel = ReferenceChannel("B+", ("pi+", "pi+", "pi-"))
    model = DecayModel(
        channel,
        [
            Resonance(
                name,
                (2, 0),
                1.0,
                lineshape=shape(),
                angular=ZemachP(),
                mass=mass,
                width=width,
                spin=spin,
                resonance_radius=4.0,
                parent_radius=4.0,
                normalize_component=False,
                normalize_form_factors=False,
            )
            for name, mass, width, spin, shape in CATALOG
        ],
        normalize_components=False,
    )

    # Our barriers are unity at the pole; Laura uses 1/sqrt(P_L(z)).
    # F_ours = sqrt(P_L(z_q0)*P_L(z_p0)) * F_Laura.
    def polynomial(z, spin):
        return (1.0, 1.0 + z, 9.0 + 3 * z + z * z, 225.0 + 45 * z + 6 * z * z + z**3)[
            spin
        ]

    factors = []
    for _, mass, _, spin, _ in CATALOG:
        q = np.sqrt(mass * mass / 4 - args.pion_mass**2)
        e = (args.parent_mass**2 - mass**2 - args.pion_mass**2) / (2 * mass)
        p = np.sqrt(e * e - args.pion_mass**2)
        factors.append(
            np.sqrt(polynomial((4 * q) ** 2, spin) * polynomial((4 * p) ** 2, spin))
        )
    factors = np.array(factors)
    efficiencies = []
    references = []
    hashes = {}
    for filename in (
        "GenFit3piCP_R2_415489.cc",
        "GenFit3piCP_fit_415489.log",
        "inputs/ACC_pipipi_15161718_NoSpike_prodAsy.root",
    ):
        hashes[filename] = hashlib.sha256(
            (args.inputs / filename).read_bytes()
        ).hexdigest()
    for charge in ("pos", "neg"):
        path = args.inputs / f"integ_{charge}.dat"
        names, bare, accepted = read_integrals(path)
        if names != [row[0] for row in CATALOG]:
            raise ValueError("Unexpected component order")
        references.append((bare, accepted))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        hist = "Hacc_pipipi_15161718_" + ("Plus" if charge == "pos" else "Minus")
        values, xe, ye = read_root_histogram2d(
            args.inputs / "inputs/ACC_pipipi_15161718_NoSpike_prodAsy.root", hist
        )
        efficiencies.append(
            SquareDalitzHistogramEfficiency(
                mprime_edges=xe,
                thetaprime_edges=ye,
                values=values,
                mother_mass=args.parent_mass,
                masses=channel.daughter_masses,
                pair=(0, 1),
                folded=True,
                interpolation="linear",
                clip=True,
            )
        )
    veto = CompositeVeto(
        MassWindowVeto((0, 2), 1.74, 1.894), MassWindowVeto((1, 2), 1.74, 1.894)
    )

    @jax.jit
    def reduce(data, weights):
        amplitudes = jnp.stack(
            [c.function(data, {}) for c in model.amplitude_model.components], axis=1
        )

        def matrix(w):
            return jnp.einsum("n,ni,nj->ij", w, amplitudes, jnp.conj(amplitudes))

        return jnp.stack(
            [matrix(weights)]
            + [matrix(weights * veto(data) * eff(data)) for eff in efficiencies]
        )

    output = {
        "catalog": [list(row[:4]) for row in CATALOG],
        "hashes": hashes,
        "parent_mass": args.parent_mass,
        "pion_mass": args.pion_mass,
        "amplitude_scale_ours_over_laura": factors.tolist(),
        "reference": [
            {"bare": encode(b), "accepted": encode(a)} for b, a in references
        ],
        "grids": [],
    }

    def grids():
        for resolution in args.resolutions:
            yield (
                str(resolution),
                SquareDalitzGrid(
                    args.parent_mass,
                    channel.daughter_masses,
                    resolution=resolution,
                    quadrature="gauss-legendre",
                ).sample(),
            )
        yield (
            "laura-logged-mass-grid",
            logged_grid(args.inputs, args.parent_mass, args.pion_mass),
        )

    for resolution, sample in grids():
        matrices = np.zeros((3, len(CATALOG), len(CATALOG)), complex)
        for start in range(0, sample.size, 50000):
            stop = min(start + 50000, sample.size)
            pad = 50000 - (stop - start)
            data = {
                k: jnp.pad(v[start:stop], (0, pad), mode="edge")
                for k, v in sample.as_dict().items()
                if k in ("s12", "s13", "s23")
            }
            weights = jnp.pad(sample.weights[start:stop] / sample.size, (0, pad))
            matrices += np.asarray(reduce(data, weights))
        row = {
            "resolution": resolution,
            "bare": encode(matrices[0]),
            "accepted_pos": encode(matrices[1]),
            "accepted_neg": encode(matrices[2]),
            "comparisons": [],
        }
        for charge, (bare, accepted), actual in zip(
            ("pos", "neg"), references, matrices[1:], strict=True
        ):
            for kind, ref, ours in (
                ("bare", bare, matrices[0]),
                ("accepted", accepted, actual),
            ):
                diagonal = np.diag(ours).real / np.diag(ref).real - 1
                normalized = np.abs(ours - ref) / np.sqrt(
                    np.outer(np.diag(ref).real, np.diag(ref).real)
                )
                row["comparisons"].append(
                    {
                        "charge": charge,
                        "kind": kind,
                        "diagonal_relative_difference": diagonal.tolist(),
                        "max_matrix_difference_over_diagonal_scale": float(
                            normalized.max()
                        ),
                        "matrix_difference_over_diagonal_scale": normalized.tolist(),
                    }
                )
                print(
                    resolution,
                    charge,
                    kind,
                    "diagonal %",
                    100 * diagonal,
                    "max scaled matrix difference",
                    normalized.max(),
                    flush=True,
                )
        output["grids"].append(row)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
