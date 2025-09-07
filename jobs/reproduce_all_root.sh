#!/usr/bin/env bash
#SBATCH --job-name=physio_gpu_seed1
#SBATCH --partition=STUD             
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/t-PatchGNN/tPatchGNN

source $HOME/venv310/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$HOME/Test/t-PatchGNN:$HOME/Test/t-PatchGNN/tPatchGNN:${PYTHONPATH-}"

python run_models.py \
  --dataset physionet --state def --history 24 \
  --patience 10 --batch_size 32 --lr 1e-3 \
  --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 64 \
  --outlayer Linear --seed 1 --gpu 0 \
  --quantization 1.0 \
  --save experiments/physio_gpu_seed1
