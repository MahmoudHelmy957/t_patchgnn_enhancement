#!/bin/bash
#SBATCH --job-name=tpatch_physio_cpu
#SBATCH --partition=TEST
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Paths
REPO="$HOME/Test/t_patchgnn_enhancement"
ENV_ACT="$HOME/venv310/bin/activate"

# Env: CPU only
source "$ENV_ACT"
unset CUDA_VISIBLE_DEVICES
unset CUDA_DEVICE_ORDER
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
# optional allocator tweak for CPU BLAS threading behavior
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

# Put temp files on scratch (avoid small /tmp)
export SCRATCH_DIR=${SCRATCH:-$HOME/scratch}/tpatchgnn
mkdir -p "$SCRATCH_DIR/tmp" "$SCRATCH_DIR/logs"
export TMPDIR="$SCRATCH_DIR/tmp"

cd "$REPO"

# Run a short CPU-friendly config to fit within 1h
python -u run_models.py \
  --dataset physionet --state def --history 48 \
  --patience 3 --batch_size 16 --lr 1e-3 \
  --patch_size 8 --stride 8 \
  --nhead 1 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 64 \
  --outlayer mlp --seed 0 --gpu -1 \
  --quantization 1.0
