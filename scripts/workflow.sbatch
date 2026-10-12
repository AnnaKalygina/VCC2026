#!/usr/bin/env bash
#SBATCH -p normal.120h
#SBATCH -c 1
#SBATCH --mem-per-cpu=4000M
#SBATCH -t 5-00:00:00
#SBATCH -J vcc_stats
set -euo pipefail
# Submit from the project root: sbatch scripts/workflow.sbatch [targets...]
cd "${SLURM_SUBMIT_DIR:?}"
test -f workflow/Snakefile
mkdir -p logs/slurm
bash scripts/run_job.sh pixi run snakemake -s workflow/Snakefile --workflow-profile "workflow/profiles/${WORKFLOW_PROFILE:-slurm}" --rerun-incomplete --keep-going "$@"
