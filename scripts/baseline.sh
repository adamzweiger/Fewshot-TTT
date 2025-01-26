#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/current/out.log
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dogo
#SBATCH --partition=long

# ====================
# Environment Setup
# ====================
export HOME=/data/cl/u/adamz
source ~/.bashrc
conda activate tttenv
cd ~/Fewshot-TTT

# ====================
# Configurable Parameters
# ====================
DATE=$(date +"%Y%m%d")
LOG_DIR="logs/current"

NUM_GPUS=1

EXP_NAME="ZSL"
MODEL_DIR="${HOME}/Models/Llama-3.1-8B-Instruct"

K_SHOT=0  # Set to 0 for zero-shot; change as needed

MAJORITY_VOTE=False
VOTE_PERMUTATIONS=5

SEED=42

# Task Range
TASK_START=0
TASK_END=27

# ====================
# Run Jobs
# ====================
TOTAL_TASKS=$((TASK_END - TASK_START))
TASKS_PER_GPU=$((TOTAL_TASKS / NUM_GPUS))
REMAINDER=$((TOTAL_TASKS % NUM_GPUS))

# Ensure log and output directories exist
mkdir -p "${LOG_DIR}"

# Array to hold PIDs of background processes
PIDS=()

for (( i=0; i<NUM_GPUS; i++ )); do
    GPU_ID=$i
    GPU_TASK_START=$((TASK_START + i * TASKS_PER_GPU))

    # Assign remainder tasks to the last GPU
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        GPU_TASK_END=$((GPU_TASK_START + TASKS_PER_GPU + REMAINDER))
    else
        GPU_TASK_END=$((GPU_TASK_START + TASKS_PER_GPU))
    fi

    OUTPUT_FILE="${LOG_DIR}/${DATE}_${EXP_NAME}_gpu${i}.json"

    echo "Launching task range [${GPU_TASK_START}, ${GPU_TASK_END}) on GPU ${GPU_ID}" > "${LOG_DIR}/${DATE}_${EXP_NAME}_gpu${i}.log"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m src.methods.baseline \
        --exp_name "${EXP_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --output_file "${OUTPUT_FILE}" \
        --task_start "${GPU_TASK_START}" \
        --task_end "${GPU_TASK_END}" \
        --k "${K_SHOT}" \
        --majority_vote "${MAJORITY_VOTE}" \
        --vote_permutations "${VOTE_PERMUTATIONS}" \
        --seed "${SEED}" \
        > "${LOG_DIR}/${DATE}_${EXP_NAME}_gpu${i}.log" 2>&1 &

    PIDS+=($!)
done

# ====================
# Wait for All Jobs to Complete
# ====================
for PID in "${PIDS[@]}"; do
    wait $PID
    if [ $? -ne 0 ]; then
        echo "A process failed. Check logs for details."
        exit 1
    fi
done

# ====================
# Combine Results
# ====================
COMBINED_OUTPUT_FILE="${LOG_DIR}/${DATE}_${EXP_NAME}.json"

echo "Combining results into ${COMBINED_OUTPUT_FILE}..."
python3 src/combine_task_results.py "${LOG_DIR}/${DATE}_${EXP_NAME}_gpu"*.json \
    --output "${COMBINED_OUTPUT_FILE}"

if [ $? -eq 0 ]; then
    echo "Results combined successfully into ${COMBINED_OUTPUT_FILE}."
    
    # ====================
    # Delete Individual GPU JSON Files
    # ====================
    echo "Deleting individual GPU JSON files..."
    rm "${LOG_DIR}/${DATE}_${EXP_NAME}_gpu"*.json
    if [ $? -eq 0 ]; then
        echo "Individual GPU JSON files deleted successfully."
    else
        echo "Failed to delete individual GPU JSON files. Please check permissions."
        exit 1
    fi
else
    echo "Failed to combine results. Check logs for details."
    exit 1
fi

# ====================
# Completion Message
# ====================
echo "Baseline evaluation completed. Combined results saved to: ${COMBINED_OUTPUT_FILE}"
