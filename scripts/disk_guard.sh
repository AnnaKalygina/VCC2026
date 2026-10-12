#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."
while kill -0 "$1" 2>/dev/null; do
    if [[ -d data/raw/_clean_tmp ]]; then
        used=$(du -sb data/raw/_clean_tmp | cut -f1)
        if (( used > 245000000000 )); then
            echo "disk_guard: $used bytes in data/raw/_clean_tmp exceeds 245 GB; cancelling this Slurm job" >&2
            scancel "${SLURM_JOB_ID:?}"
            exit 1
        fi
    fi
    sleep 60
done
