#!/bin/bash
set -e  # stop script on error

for i in {1..24}
do
    echo "Run #$i at $(date)"

    # Check if there are changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        git add .
        git commit -m "Added results at $(date '+%Y-%m-%d %H:%M:%S')"
        git push origin main
    else
        echo "No changes to commit"
    fi

    # sleep 30 minutes (1800 seconds), but not after last run
    if [ "$i" -lt 24 ]; then
        sleep 1800
    fi
done