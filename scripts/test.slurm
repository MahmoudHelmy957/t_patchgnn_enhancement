#!/bin/bash
#SBATCH --job-name=tpatch_physio_test
#SBATCH --partition=TEST
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# === paths ===
REPO="$HOME/Test/t_patchgnn_enhancement"
ENV_ACT="$HOME/venv310/bin/activate"

# === env ===
source "$ENV_ACT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # helps avoid OOM fragmenting

# (optional) put temp files on scratch
export SCRATCH_DIR=${SCRATCH:-$HOME/scratch}/tpatchgnn
mkdir -p "$SCRATCH_DIR/tmp" "$SCRATCH_DIR/logs"
export TMPDIR="$SCRATCH_DIR/tmp"

cd "$REPO"

# quick context
hostname
nvidia-smi || true
python -c "import torch;print('CUDA?',torch.cuda.is_available(),'GPUs',torch.cuda.device_count())"

# === run ===
python -u run_models.py \
  --dataset physionet --state def --history 48 \
  --patience 3 --batch_size 16 --lr 1e-3 \
  --patch_size 8 --stride 8 \
  --nhead 1 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 64 \
  --outlayer mlp --seed 0 --gpu 0 \
  --quantization 1.0