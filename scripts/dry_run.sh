#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."
export XDG_CACHE_HOME="$PWD/.cache"
snakemake -s workflow/Snakefile -n --cores 1 "$@"
