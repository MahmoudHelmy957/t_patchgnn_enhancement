#!/usr/bin/env bash
#SBATCH --job-name=tiny
#SBATCH --output=%j.log
#SBATCH --error=%j.err
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1       # navigate to the directory if necessary


srun python -c "import torch; print(torch.cuda.is_available())"