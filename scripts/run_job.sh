#!/usr/bin/env bash
# All paths are relative to this copy, independently of the caller's directory.
set -euo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."
: "${SLURM_JOB_ID:?Run data work through Slurm}"
. "${LMOD_ROOT:?}/lmod/init/bash"
ml eth_proxy
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
bash scripts/disk_guard.sh $$ &
guard=$!
trap 'kill "$guard" 2>/dev/null || true' EXIT
"$@"
