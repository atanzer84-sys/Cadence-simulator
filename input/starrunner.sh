#!/bin/bash
set -u

SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/src/cadence_simulator.py"

if [ $# -ge 1 ]; then
    BASE_DIR="$1"
    if compgen -G "$BASE_DIR"/*.txt > /dev/null; then
        DIRS=("$BASE_DIR")
    else
        DIRS=("$BASE_DIR"/*/)
    fi
else
    DIRS=(*/)
fi

for dir in "${DIRS[@]}"; do
    [ -d "$dir" ] || continue
    for file in "$dir"/*.txt; do
        [ -f "$file" ] || continue
        echo "=================================================="
        echo "RUNNING: $file"
        echo "=================================================="
        python "$SCRIPT" "$file"
        status=$?
        echo "EXIT CODE: $status"
        if [ $status -ne 0 ]; then
            echo "FAILED: $file"
        else
            echo "FINISHED: $file"
        fi
        echo
    done
done