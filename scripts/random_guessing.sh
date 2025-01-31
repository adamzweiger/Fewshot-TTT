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

EXP_NAME="random_guessing"

# Task Range
TASK_START=0
TASK_END=27

# ====================
# Ensure Log Directory Exists
# ====================
mkdir -p "${LOG_DIR}"

OUTPUT_FILE="${LOG_DIR}/${DATE}_${EXP_NAME}.json"

# ====================
# Run Random Guessing Job
# ====================
echo "Launching random guessing from task ${TASK_START} to ${TASK_END}..." > "${LOG_DIR}/${DATE}_${EXP_NAME}.log"

CUDA_VISIBLE_DEVICES=0 python3 -m src.methods.random_guessing \
    --exp_name "${EXP_NAME}" \
    --output_file "${OUTPUT_FILE}" \
    --task_start "${TASK_START}" \
    --task_end "${TASK_END}" \
    >> "${LOG_DIR}/${DATE}_${EXP_NAME}.log" 2>&1

# ====================
# Check Execution Status
# ====================
if [ $? -eq 0 ]; then
    echo "Random guessing completed successfully. Results saved to: ${OUTPUT_FILE}" >> "${LOG_DIR}/${DATE}_${EXP_NAME}.log"
else
    echo "Random guessing failed. Check logs for details." >> "${LOG_DIR}/${DATE}_${EXP_NAME}.log"
    exit 1
fi

# ====================
# Completion Message
# ====================
echo "Random guessing job finished." 

