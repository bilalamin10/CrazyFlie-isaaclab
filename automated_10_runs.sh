#!/bin/bash

# --- CONFIGURATION ---
NUM_RUNS=3
TASK_NAME="Isaac-Quadcopter-Direct-v0"
SCRIPT_PATH="scripts/rsl_rl/train.py"

# --- FIXED SEED (or change this if you want different seeds) ---
FIXED_SEED=111 
# Experiment Name will be the main folder for grouping all 10 runs. 
# We'll use a simple group name here.
GROUP_NAME="Sweep_10x_SimpleLog"

echo "Starting ${NUM_RUNS} runs using FIXED SEED ${FIXED_SEED}..."

for i in $(seq 1 $NUM_RUNS); do
    # Define a unique run name using the counter 'i'
    RUN_NAME="Run_${i}"

    echo "--- Launching Trial $i/$NUM_RUNS ---"
    
    # Execute the training command:
    # We use the simple, direct flags that have high precedence.
    python ${SCRIPT_PATH} --task ${TASK_NAME} --headless \
        --num_envs 4096 \
        --seed ${FIXED_SEED} \
        --experiment_name ${GROUP_NAME} \
        --run_name ${RUN_NAME}
        
    sleep 3
done

echo "All ${NUM_RUNS} runs complete. Check the 'logs/${GROUP_NAME}' directory."