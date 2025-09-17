#!/usr/bin/env bash
#SBATCH --job-name=physio_STUD_ms_2_8_24_ol
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --chdir=/home/ouass/Test/t-PatchGNN/logs

set -euo pipefail
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$HOME/Test/t-PatchGNN:$HOME/Test/t-PatchGNN/tPatchGNN:${PYTHONPATH-}"

# run from code dir
cd "$HOME/Test/t-PatchGNN/tPatchGNN"

SEED=${SLURM_ARRAY_TASK_ID}
GPU=0
EPOCHS=600
PATIENCE=60
BATCH=32          # if OOM, drop to 28 or 24
LR=1e-3
HISTORY=24
QUANT=1.0

# 3-scale with overlap: (patch/stride) = (2/1, 8/4, 24/12)
SCALES="2,8,24"
STRIDES="1,4,12"
FUSION="concat"   # change to 'scale_attn' to try attention fusion

echo "STUD MS 3-scale overlap: seed=$SEED scales=$SCALES strides=$STRIDES fusion=$FUSION"

python run_models.py \
  --dataset physionet \
  --history $HISTORY \
  --quantization $QUANT \
  --hid_dim 64 \
  --te_dim 10 \
  --node_dim 10 \
  --nlayer 1 \
  --tf_layer 1 \
  --nhead 1 \
  --batch_size $BATCH \
  --lr $LR \
  --patience $PATIENCE \
  --epoch $EPOCHS \
  --seed $SEED \
  --gpu $GPU \
  --multi_scales "$SCALES" \
  --multi_strides "$STRIDES" \
  --fusion "$FUSION"
