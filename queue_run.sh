#!/bin/bash
# queue_run.sh
# Usage: ./queue_run.sh "your python command here"
# Example: ./queue_run.sh "python scripts/unlearn.py --methods rase"

COMMAND=$1
# Minimum free memory required in MiB (e.g., 25000 = ~25GB)
MIN_MEM=25000

echo "Waiting for $MIN_MEM MiB free GPU memory..."
while true; do
    # Get free memory of GPU 0
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    
    if [ "$FREE_MEM" -ge "$MIN_MEM" ]; then
        echo "Found $FREE_MEM MiB free! Starting task..."
        eval $COMMAND
        break
    fi
    
    # Wait 60 seconds before checking again
    sleep 60
done
