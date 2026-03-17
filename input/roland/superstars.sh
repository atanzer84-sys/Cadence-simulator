#!/bin/bash
set -u

PYTHON="/Users/andreatanzer/.venvs/shared/bin/python"
SCRIPT="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/src/waltzer_simulator.py"

for dir in */; do
    [ -d "$dir" ] || continue

    for file in "$dir"/*.txt; do
        [ -f "$file" ] || continue

        echo "=================================================="
        echo "RUNNING: $file"
        echo "=================================================="

        "$PYTHON" "$SCRIPT" "$file"
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