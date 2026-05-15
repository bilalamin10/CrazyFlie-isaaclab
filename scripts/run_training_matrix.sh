#!/bin/bash
# Training matrix: 5 shapes × 3 seeds × 200 iters = 15 runs
# Each run goes to logs/rsl_rl/<Shape>_seed<N>/<timestamp>/

set -e  # Stop on first error

SHAPES=("Hover" "Static" "Circle" "Lemniscate" "Lissajous")
SEEDS=(0 1 2)
ITERS=400
NUM_ENVS=4096

TOTAL=$((${#SHAPES[@]} * ${#SEEDS[@]}))
COUNT=0

echo "=============================================="
echo "Baseline Training Matrix"
echo "Shapes: ${SHAPES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Iterations per run: ${ITERS}"
echo "Num envs: ${NUM_ENVS}"
echo "Total runs: ${TOTAL}"
echo "Started: $(date)"
echo "=============================================="

for SHAPE in "${SHAPES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        EXP_NAME="${SHAPE}_seed${SEED}"
        echo ""
        echo "=== [${COUNT}/${TOTAL}] [$(date +%H:%M:%S)] Training ${EXP_NAME} ==="
        python scripts/rsl_rl/train.py \
            --task "Isaac-Quadcopter-${SHAPE}-Direct-v0" \
            --num_envs ${NUM_ENVS} \
            --max_iterations ${ITERS} \
            --seed ${SEED} \
            --experiment_name "${EXP_NAME}" \
            --headless
    done
done

echo ""
echo "=============================================="
echo "All ${TOTAL} training runs complete!"
echo "Finished: $(date)"
echo "=============================================="