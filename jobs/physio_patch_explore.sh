#!/usr/bin/env bash
#SBATCH --job-name=physio_patch_explore
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/t-PatchGNN/logs

set -euo pipefail
source $HOME/venv310/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$HOME/Test/t-PatchGNN:$HOME/Test/t-PatchGNN/tPatchGNN:${PYTHONPATH-}"

cd $HOME/Test/t-PatchGNN/tPatchGNN

patience=10
gpu=0

for ps in 12 10 14; do
  for stride in 4 5 7; do
    # match only valid combos: (12,4), (10,5), (14,7)
    if [[ ($ps -eq 12 && $stride -eq 4) || ($ps -eq 10 && $stride -eq 5) || ($ps -eq 14 && $stride -eq 7) ]]; then
      for seed in {1..5}; do
        python run_models.py \
          --dataset physionet --state 'def' --history 24 \
          --patience $patience --batch_size 32 --lr 1e-3 \
          --patch_size $ps --stride $stride --nhead 1 --tf_layer 1 --nlayer 1 \
          --te_dim 10 --node_dim 10 --hid_dim 64 \
          --outlayer Linear --seed $seed --gpu $gpu \
          --quantization 1.0 \
          --save experiments/physio_ps${ps}_s${stride}_seed${seed}
      done
    fi
  done
done
