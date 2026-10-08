#!/usr/bin/env bash
# Cancel the job if the staged-download area grows past DISK_GUARD_BYTES (default 245 GB, Euler's
# scratch budget; 0 disables). Only data/raw/_clean_tmp is scanned: a whole-repo du over retained raw
# data on CephFS every minute is needlessly expensive.
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
