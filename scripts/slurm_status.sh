#!/usr/bin/env bash
set -euo pipefail
s=$(sacct -j "$1" -X -n -o State | head -1 | tr -d ' ')
case "$s" in
    COMPLETED) echo success ;;
    PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|CONFIGURING|COMPLETING) echo running ;;
    "") echo "WARNING no scheduler accounting state for $1; treating as running" >&2; echo running ;;
    *) echo failed ;;
esac
