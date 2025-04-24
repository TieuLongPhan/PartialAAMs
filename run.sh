#!/bin/bash

LOG_FILE="./Data/batch_run.log"
# Clear out previous log content
> "$LOG_FILE"

echo "Starting batch run at $(date)" | tee -a "$LOG_FILE"
START_TOTAL=$(date +%s)

# Define an array of scripts to run
scripts=("benchmarking.py")

# Loop over each script
for script in "${scripts[@]}"; do
    echo "============================================" | tee -a "$LOG_FILE"
    echo "Running script: $script" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    
    # Run 5 iterations for the current script
    for i in {1..5}; do
        echo "-> [$script] Starting iteration $i at $(date)" | tee -a "$LOG_FILE"
        ITER_START=$(date +%s)
        
        # Run the python script
        python "$script"
        EXIT_CODE=$?
        
        ITER_END=$(date +%s)
        DURATION=$((ITER_END - ITER_START))
        
        if [ $EXIT_CODE -ne 0 ]; then
            echo "!! [$script] Iteration $i FAILED with exit code $EXIT_CODE after ${DURATION}s" | tee -a "$LOG_FILE"
        else
            echo "++ [$script] Iteration $i completed successfully in ${DURATION}s" | tee -a "$LOG_FILE"
        fi
        echo "--------------------------------------------" | tee -a "$LOG_FILE"
    done
done

END_TOTAL=$(date +%s)
TOTAL_DURATION=$((END_TOTAL - START_TOTAL))
echo "All iterations completed in ${TOTAL_DURATION}s at $(date)" | tee -a "$LOG_FILE"
