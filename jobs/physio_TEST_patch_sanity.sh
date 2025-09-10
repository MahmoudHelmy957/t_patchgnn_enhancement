#!/usr/bin/env bash
#SBATCH --job-name=physio_TEST_patch_sanity
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000
#SBATCH --time=00:59:00
#SBATCH --array=0-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --chdir=/home/ouass/Test/t-PatchGNN/logs

set -euo pipefail
source $HOME/venv310/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$HOME/Test/t-PatchGNN:$HOME/Test/t-PatchGNN/tPatchGNN:${PYTHONPATH-}"

cd $HOME/Test/t-PatchGNN/tPatchGNN

# grid: (patch_size, stride)
PS=(8 8 8 12)
ST=(8 6 4 6)

i=${SLURM_ARRAY_TASK_ID}
ps=${PS[$i]}
st=${ST[$i]}

echo "Running TEST sanity: ps=$ps stride=$st seed=1"

python run_models.py \
  --dataset physionet --state def --history 24 \
  --patience 10 --batch_size 32 --lr 1e-3 \
  --patch_size $ps --stride $st --nhead 1 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 64 \
  --outlayer Linear --seed 1 --gpu 0 \
  --quantization 1.0 \
  --save experiments/physio_ps${ps}_s${st}_seed1_TEST
