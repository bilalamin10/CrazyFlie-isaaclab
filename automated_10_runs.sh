#!/bin/bash

# Configuration
SEED_VALUE=111      # The seed you want to test (e.g., 111)
NUM_RUNS=10         # Total number of repeated experiments
TASK_NAME="Isaac-Quadcopter-Direct-v0"
CHECKPOINT_PATH="/home/vikas-group3/IsaacLab/isaaclab.sh"
SCRIPT_PATH="scripts/reinforcement_learning/rsl_rl/train.py"

echo "Starting ${NUM_RUNS} training runs for SEED ${SEED_VALUE}..."

for i in $(seq 1 $NUM_RUNS); do
    # Define a unique experiment name for each run:
    EXPERIMENT_NAME="Seed${SEED_VALUE}_Run${i}"
    
    echo "--- Launching $EXPERIMENT_NAME (Run $i/$NUM_RUNS) ---"
    
    # Execute the training command:
    # 1. Use --seed to set the fixed seed for all runs (111).
    # 2. Use runner.experiment_name to get a unique log folder name.
    # 3. Use runner.run_name to include the run number for finer logging distinction.
    
    # NOTE: The runner.experiment_name MUST be different for each run to avoid overwriting.
    
    ${CHECKPOINT_PATH} -p ${SCRIPT_PATH} --task ${TASK_NAME} --headless \
        --seed ${SEED_VALUE} \
        runner.experiment_name="${EXPERIMENT_NAME}" \
        runner.run_name="TRIAL_${i}"
        
    echo "--- Finished ${EXPERIMENT_NAME} ---"
done

echo "All ${NUM_RUNS} runs complete. Check your 'logs' directory."