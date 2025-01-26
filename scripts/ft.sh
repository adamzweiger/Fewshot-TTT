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
source ~/.bashrc~
conda activate tttenv
cd ~/Fewshot-TTT

# ====================
# Configurable Parameters
# ====================
DATE=$(date +"%Y%m%d")
LOG_DIR="logs/current"

NUM_GPUS=1

EXP_NAME="FT_10_5_4_1e-4_64_64_0.05"
# Naming: FT_k_batchSize_epochs_lr_loraRank_loraAlpha_loraDropout
MODEL_DIR="${HOME}/Models/Llama-3.1-8B-Instruct"

K_SHOT=10

# Finetuning Hyperparameters
BATCH_SIZE=5
EPOCHS=4
LR=1e-4
LORA_R=64
LORA_ALPHA=64
LORA_DROPOUT=0.05

SEED=42

# Task Range
TASK_START=0
TASK_END=27

# Output Files
OUTPUT_FILE_PREFIX="${DATE}_${EXP_NAME}_gpu"

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

    OUTPUT_FILE="${LOG_DIR}/${OUTPUT_FILE_PREFIX}${i}.json"

    echo "Launching task range [${GPU_TASK_START}, ${GPU_TASK_END}) on GPU ${GPU_ID}" > "${LOG_DIR}/${OUTPUT_FILE_PREFIX}${i}.log"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m src.methods.ft \
        --exp_name "${EXP_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --output_file "${OUTPUT_FILE}" \
        --task_start "${GPU_TASK_START}" \
        --task_end "${GPU_TASK_END}" \
        --k "${K_SHOT}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --lora_rank "${LORA_R}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --seed "${SEED}" \
        > "${LOG_DIR}/${OUTPUT_FILE_PREFIX}${i}.log" 2>&1 &

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
python3 src/combine_task_results.py "${LOG_DIR}/${OUTPUT_FILE_PREFIX}"*.json \
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
echo "Finetuning completed. Combined results saved to: ${COMBINED_OUTPUT_FILE}"
