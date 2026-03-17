#!/bin/bash
set -u

PYTHON="/Users/andreatanzer/.venvs/shared/bin/python"
SCRIPT="/Users/andreatanzer/Documents/Space Science/MasterThesis/WALTzER-simulator/src/waltzer_simulator.py"

# if argument given → only that dir, else all dirs
if [ $# -ge 1 ]; then
    DIRS=("$1")
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


#./script.sh              # all folders
# ./script.sh 5800_11_G/  # just one folder