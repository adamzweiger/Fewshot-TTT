#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/current2/shared_e2e_%A_%a.log
#SBATCH --nodelist=dogo
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --array=0-4

# ====================
# Environment Setup
# ====================
export HOME=/data/cl/u/adamz
source ~/.bashrc
conda activate tttenv
cd ~/Fewshot-TTT

# -------------------------------------------------------------------------
# Fixed/Shared Parameters
# -------------------------------------------------------------------------
DATE=$(date +"%Y%m%d")
LOG_DIR="logs/current"
MODEL_DIR="${HOME}/Models/Llama-3.1-8B-Instruct"

# Example: number of examples per task for finetuning (k-shot)
K_SHOT=10

mkdir -p "${LOG_DIR}"

# -------------------------------------------------------------------------
# Experiment parameter sets
# Each entry typically:
#   "exp LR BATCH_SIZE EPOCHS LORA_RANK LORA_ALPHA LORA_DROPOUT SEED"
# -------------------------------------------------------------------------
EXPERIMENTS=( # 5e-5, 1e-4; 5, 10; 2, 4, 6; 64, 128; 64, 128; 0.05; 42
  "exp 5e-5 5 6 64 64 0.05 42"
  "exp 5e-5 5 6 64 64 0.05 43"
  "exp 5e-5 5 6 64 64 0.05 44"
  "exp 5e-5 5 6 64 64 0.05 45"
  "exp 5e-5 5 6 64 64 0.05 46"
)

# -------------------------------------------------------------------------
# Select parameters for THIS sub-job using SLURM_ARRAY_TASK_ID
# -------------------------------------------------------------------------
PARAMS="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
read -r EXP_NOTE LR BATCH_SIZE EPOCHS LORA_RANK LORA_ALPHA LORA_DROPOUT SEED <<< "${PARAMS}"

# Construct a descriptive experiment name:
#   SHARED_<EXP_NOTE>_<K_SHOT>_<BATCH_SIZE>_<EPOCHS>_<LR>_<LORA_RANK>_<LORA_ALPHA>_<LORA_DROPOUT>_<SEED>
EXP_NAME="SHARED_E2E_${EXP_NOTE}_${K_SHOT}_${BATCH_SIZE}_${EPOCHS}_${LR}_${LORA_RANK}_${LORA_ALPHA}_${LORA_DROPOUT}_${SEED}"

# Output filenames
OUTPUT_FILE="${LOG_DIR}/${DATE}_${EXP_NAME}.json"
LOGFILE="${LOG_DIR}/${DATE}_${EXP_NAME}.log"

echo "------------------------------------------------------------"
echo "[SUB-JOB: $SLURM_ARRAY_TASK_ID] Launching Shared E2E Experiment:"
echo "  EXP_NAME = ${EXP_NAME}"
echo "------------------------------------------------------------"

# -------------------------------------------------------------------------
# Run the Python finetuning code (shared E2E)
# -------------------------------------------------------------------------
python3 -m src.methods.shared_e2e \
  --exp_name "${EXP_NAME}" \
  --model_dir "${MODEL_DIR}" \
  --output_file "${OUTPUT_FILE}" \
  --k "${K_SHOT}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --seed "${SEED}" \
  > "${LOGFILE}" 2>&1

echo "Shared E2E sub-job $SLURM_ARRAY_TASK_ID completed."
