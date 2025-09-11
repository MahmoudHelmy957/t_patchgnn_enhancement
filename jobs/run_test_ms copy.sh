#!/usr/bin/env bash
#SBATCH --job-name=physio_TEST_ms_full
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000
#SBATCH --time=00:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/t-PatchGNN/logs

set -euo pipefail

# venv
source "$HOME/venv310/bin/activate"

# threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# repo path
export PYTHONPATH="$HOME/Test/t-PatchGNN:$HOME/Test/t-PatchGNN/tPatchGNN:${PYTHONPATH-}"

# go to code dir
cd "$HOME/Test/t-PatchGNN/tPatchGNN"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -V

# TEST-friendly knobs (full data; fewer epochs to fit 1h)
SEED=0
GPU=0
EPOCHS=30          
PATIENCE=10
BATCH=32           
LR=1e-3
HISTORY=24
QUANT=1.0

# Multi-scale: 2h + 8h (no overlap)
SCALES="2,8"
STRIDES="2,8"

echo "Running TEST MS FULL: scales=$SCALES strides=$STRIDES seed=$SEED (full dataset)"

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
  --fusion concat
# no -n => full 12k samples
