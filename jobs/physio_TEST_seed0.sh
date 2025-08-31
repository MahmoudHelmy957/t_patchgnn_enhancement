#!/usr/bin/env bash
#SBATCH --job-name=physio_TEST_seed0
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000
#SBATCH --time=00:59:00
#SBATCH --output=%x_.out
#SBATCH --error=%x_.err
#SBATCH --chdir=/home/ouass/Test/t-PatchGNN/logs

set -euo pipefail
source $HOME/venv310/bin/activate

export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Make both the repo root and tPatchGNN visible to Python
export PYTHONPATH="$HOME/Test/t-PatchGNN:$HOME/Test/t-PatchGNN/tPatchGNN:${PYTHONPATH-}"

# 👇 Change CWD to tPatchGNN so '../data/physionet' points to repo_root/data/physionet
cd $HOME/Test/t-PatchGNN/tPatchGNN

python run_models.py \
  --dataset physionet \
  --history 24 \
  -ps 8 --stride 8 \
  --nhead 1 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 64 \
  --batch_size 32 --lr 1e-3 --patience 10 \
  --outlayer Linear --seed 0 --gpu 0 \
  --quantization 1.0
