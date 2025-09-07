#!/usr/bin/env bash
#SBATCH --job-name=physio_STUD_seeds
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --array=1-5                   # use 1-5 to mirror your script; 0-4 also fine
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --chdir=/home/ouass/Test/t-PatchGNN/logs

set -euo pipefail
source $HOME/venv310/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$HOME/Test/t-PatchGNN:$HOME/Test/t-PatchGNN/tPatchGNN:${PYTHONPATH-}"

cd $HOME/Test/t-PatchGNN/tPatchGNN
seed=${SLURM_ARRAY_TASK_ID}

python run_models.py \
  --dataset physionet --state 'def' --history 24 \
  --patience 10 --batch_size 32 --lr 1e-3 \
  --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 64 \
  --outlayer Linear --seed $seed --gpu 0 \
  --save experiments/physio_ps8_s8_linear_seed${seed}
