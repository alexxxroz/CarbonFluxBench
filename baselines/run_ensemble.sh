#!/bin/bash
#SBATCH --account=kumarv
#SBATCH --job-name=ensemble
#SBATCH --output=logs/ensemble_%A_%a.txt
#SBATCH --error=logs/ensemble_err_%A_%a.txt
#SBATCH --time=03:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-39 

source ~/.bashrc
conda activate ct-lstm

MODELS=("ctgru") #("lstm" "gru" "ctlstm" "transformer" "patch_transformer")
SPLITS=("IGBP" "Koppen")
FEATURES=("standard" "full")
SEEDS=(27 28 29 30 31 32 33 34 35 36)

NUM_MODELS=${#MODELS[@]}
NUM_SPLITS=${#SPLITS[@]}
NUM_FEATURES=${#FEATURES[@]}
NUM_SEEDS=${#SEEDS[@]}

SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
TEMP1=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

FEATURE_IDX=$((TEMP1 % NUM_FEATURES))
TEMP2=$((TEMP1 / NUM_FEATURES))

SPLIT_IDX=$((TEMP2 % NUM_SPLITS))
MODEL_IDX=$((TEMP2 / NUM_SPLITS))

MODEL=${MODELS[$MODEL_IDX]}
SPLIT=${SPLITS[$SPLIT_IDX]}
FEATURE=${FEATURES[$FEATURE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

BATCH_SIZE=2048
EPOCHS=100
echo "Job ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Split: $SPLIT"
echo "Seed: $SEED"

echo "Feature set: $FEATURE"

python train_single.py \
    --model $MODEL \
    --split_type $SPLIT \
    --feature_set $FEATURE \
    --seed $SEED \
    --config ./configs/${MODEL}_${SPLIT}.yaml \
    --output_dir ./models \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --device cuda