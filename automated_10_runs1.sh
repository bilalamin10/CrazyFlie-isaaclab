#!/bin/bash

# --- CONFIGURATION ---
NUM_RUNS=2  # Number of tests per trajectory
TASK_NAME="Isaac-Quadcopter-Direct-v0"
SCRIPT_PATH="scripts/rsl_rl/play.py"
TRAJECTORIES=("circle")
FIXED_SEED=333 

echo "Starting Evaluation Sweep..."

# Outer Loop: Trajectory Types
for TRAJ in "${TRAJECTORIES[@]}"; do
    echo "========================================"
    echo "TESTING TRAJECTORY: $TRAJ"
    echo "========================================"

    # Inner Loop: Multiple runs for Variance
    for i in $(seq 1 $NUM_RUNS); do
        # 1. Create the unique identifier for this specific file
        # This is what os.getenv("RUN_NAME") will pick up in Python
        export RUN_NAME="${TRAJ}_run_${i}"

        echo "--- Launching $RUN_NAME (Seed ${FIXED_SEED}) ---"
        
        # 2. Execute play.py
        # We pass the trajectory type via the --env.trajectory_type flag
        python ${SCRIPT_PATH} --task ${TASK_NAME} --headless \
            --num_envs 64 \
            --seed ${FIXED_SEED} \
            --env.trajectory_type=${TRAJ}
            
        sleep 2
    done
done

echo "All evaluations complete. Your .npy files are ready for analysis."