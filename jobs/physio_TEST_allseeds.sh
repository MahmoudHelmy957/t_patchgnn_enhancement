#!/usr/bin/env bash
#SBATCH --job-name=physio_TEST_allseeds
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000           
#SBATCH --time=00:59:00       
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

for seed in {1..5}; do
  echo "=== Seed $seed ==="
  python run_models.py \
    --dataset physionet --state 'def' --history 24 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu \
    --quantization 1.0 \
    --save experiments/physio_ps8_s8_linear_seed${seed}
done
