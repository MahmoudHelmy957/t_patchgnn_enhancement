#!/bin/bash
#SBATCH --job-name=tpatch_physio_test
#SBATCH --partition=TEST
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# === paths (edit if your repo lives elsewhere) ===
REPO="$HOME/Test/t_patchgnn_enhancement"
ENV_ACT="$HOME/venv310/bin/activate"

# === env ===
source "$ENV_ACT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# (optional) make tmp on scratch to avoid /tmp issues
export SCRATCH_DIR=${SCRATCH:-$HOME/scratch}/tpatchgnn
mkdir -p "$SCRATCH_DIR/tmp" "$SCRATCH_DIR/logs"
export TMPDIR="$SCRATCH_DIR/tmp"

# === run ===
cd "$REPO"

# Quick CPU smoke test (keeps it under 1h):
# - smaller patience so it stops early
# - batch size 16 to reduce RAM
# - gpu -1 => CPU
python -u run_models.py \
  --dataset physionet --state def --history 48 \
  --patience 3 --batch_size 16 --lr 1e-3 \
  --patch_size 8 --stride 8 \
  --nhead 1 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 64 \
  --outlayer mlp --seed 0 --gpu -1
