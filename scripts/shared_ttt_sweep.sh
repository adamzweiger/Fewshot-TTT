#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/current/shared_ttt_%A_%a.log
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
# Fixed parameters
# -------------------------------------------------------------------------
K_SHOT=10
VOTE_PERMUTATIONS=5
LORA_DROPOUT=0.05

DATE=$(date +"%Y%m%d")
LOG_DIR="logs/current"
MODEL_DIR="${HOME}/Models/Llama-3.1-8B-Instruct"
DATASET_TYPE="masked_inputs_text_completion_dataset"

mkdir -p "${LOG_DIR}"

# -------------------------------------------------------------------------
# Experiment parameter sesqts
# (example placeholders; adapt as you like)
# -------------------------------------------------------------------------
EXPERIMENTS=( # 5e-5, 1e-4, 3e-4; 64, 128; 64, 128; 5; 1; 20, 40, 60; True; False; False; 42
  "exp 5e-5 64 64 5 1 40 True False False 42"
  "exp 5e-5 64 64 5 1 40 True False False 43"
  "exp 5e-5 64 64 5 1 40 True False False 44"
  "exp 5e-5 64 64 5 1 40 True False False 45"
  "exp 5e-5 64 64 5 1 40 True False False 46"
)

# -------------------------------------------------------------------------
# Select parameter set for THIS array sub-job
# -------------------------------------------------------------------------
PARAMS="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
read -r EXP_NOTE LR LORA_RANK LORA_ALPHA BATCH_SIZE EPOCHS NUM_TRAINING_STEPS \
       SHUFFLE MAJORITY_VOTE LEAVE_ONE_OUT SEED <<< "${PARAMS}"

# Convert SEED if desired (an example offset)
SEED_NUM=$(( SEED - 41 ))

# Construct a nice experiment name
EXP_NAME="SHARED_TTT_${EXP_NOTE}_${K_SHOT}_${DATASET_TYPE}_${NUM_TRAINING_STEPS}_${SHUFFLE}_${MAJORITY_VOTE}_${VOTE_PERMUTATIONS}_${LEAVE_ONE_OUT}_${BATCH_SIZE}_${EPOCHS}_${LR}_${LORA_RANK}_${LORA_ALPHA}_${LORA_DROPOUT}_${SEED_NUM}"
OUTPUT_FILE="${LOG_DIR}/${DATE}_${EXP_NAME}.json"
LOGFILE="${LOG_DIR}/${DATE}_${EXP_NAME}.log"

echo "----------------------------------------------------------"
echo "[SUB-JOB: $SLURM_ARRAY_TASK_ID] Launching Shared TTT Experiment:"
echo "  EXP_NAME = ${EXP_NAME}"
echo "----------------------------------------------------------"

# -------------------------------------------------------------------------
# Run the Shared TTT Python script
# -------------------------------------------------------------------------
python3 -m src.methods.shared_ttt \
  --exp_name "${EXP_NAME}" \
  --model_dir "${MODEL_DIR}" \
  --output_file "${OUTPUT_FILE}" \
  --k "${K_SHOT}" \
  --dataset_type "${DATASET_TYPE}" \
  --num_training_steps "${NUM_TRAINING_STEPS}" \
  --shuffle "${SHUFFLE}" \
  --majority_vote "${MAJORITY_VOTE}" \
  --vote_permutations "${VOTE_PERMUTATIONS}" \
  --leave_one_out "${LEAVE_ONE_OUT}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --seed "${SEED}" \
  > "${LOGFILE}" 2>&1
