#!/usr/bin/env bash
#
# Generate an ensemble of independent Laura++ toy Dalitz-plot samples for the
# B+- -> pi+ pi+ pi- CP-violating isobar model (GenFit3pi.cc), one ROOT file
# per toy, each with its own random seed, so they can be fitted externally.
#
# Usage: ./run_3pi_toys.sh [nToys=100] [firstToy=0] [baseSeed=1]
#
# Output: gen-3pi_toy<firstToy>.root ... gen-3pi_toy<firstToy+nToys-1>.root
# Tree name inside each file: genResults

set -euo pipefail

N_TOYS=${1:-100}
FIRST_TOY=${2:-0}
BASE_SEED=${3:-1}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GENFIT3PI="${SCRIPT_DIR}/GenFit3pi"

if [[ ! -x "${GENFIT3PI}" ]]; then
    echo "ERROR: ${GENFIT3PI} not found or not executable. Build it first." >&2
    exit 1
fi

for (( i=0; i<N_TOYS; i++ )); do
    toy=$(( FIRST_TOY + i ))
    seed=$(( BASE_SEED + toy ))
    echo ">>> Generating toy ${toy} (seed=${seed})"
    "${GENFIT3PI}" gen 1 "${toy}" "${seed}"
done
