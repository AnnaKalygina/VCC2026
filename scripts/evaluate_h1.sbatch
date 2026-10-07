#!/usr/bin/env bash
#SBATCH -p normal.24h
#SBATCH -c 4
#SBATCH --mem-per-cpu=8000M
#SBATCH -t 24:00:00
#SBATCH -J h1_internal
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
test -f workflow/Snakefile
mkdir -p results logs
/usr/bin/time -v bash scripts/run_job.sh pixi run python -u src/evaluate_h1.py "$@"
