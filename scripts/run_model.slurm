#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=physionet_run
=======
#SBATCH --job-name=physio_run
>>>>>>> mahmoud_code_understanding
#SBATCH --output=logs/physionet_run_%j.out
#SBATCH --error=logs/physionet_run_%j.err
#SBATCH --partition=TEST              # Specify the correct partition (not stud)
#SBATCH --gres=gpu:0                  # No GPU requested (gpu=0), change to 1 if needed

<<<<<<< HEAD
# Fix: Add lib to PYTHONPATH



=======
>>>>>>> mahmoud_code_understanding

gpu="1"



python run_models.py \
--dataset physionet \
--state 'def' \
<<<<<<< HEAD
--history 24 \
--patience 10 \
--batch_size 32 \
=======
--history 12 \
--patience 10 \
--batch_size 24 \
>>>>>>> mahmoud_code_understanding
--lr 1e-3 \
--patch_size 8 \
--stride 4 \
--nhead 1 \
--tf_layer 1 \
--nlayer 1 \
--te_dim 10 \
--node_dim 10 \
<<<<<<< HEAD
--hid_dim 64 \
--outlayer Linear \
--gpu $gpu
#--seed $seed \

=======
--hid_dim 32 \
--outlayer Linear \
--gpu $gpu
#--seed $seed \
>>>>>>> mahmoud_code_understanding
