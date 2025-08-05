# pyright: reportOptionalSubscript=false
# pyright: reportPossiblyUnboundVariable=false
import os
import sys
sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
# import random
from random import SystemRandom
# from sklearn import model_selection

import torch
# import torch.nn as nn
import torch.optim as optim

import lib.utils as utils
from lib.parse_datasets import parse_datasets
from tPatchGNN.new_model.tpatchgnn import tPatchGNN   # <-- new modular wrapper
from lib.evaluation import compute_all_losses,evaluation

parser = argparse.ArgumentParser('IMTS Forecasting')

parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--hop', type=int, default=1, help="hops in GNN")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--tf_layer', type=int, default=1, help="# of layer in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layers in encoder")
parser.add_argument('--epoch', type=int, default=1000, help="training epochs")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="length of historical window")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="stride for patch sliding")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')

parser.add_argument('--lr',  type=float, default=1e-3, help="learning rate")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay")
parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="Experiment ID to load for evaluation")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='physionet',
                    help="Dataset: physionet, mimic, ushcn")

# Quantization granularity
parser.add_argument('--quantization', type=float, default=0.0,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='tPatchGNN', help="Model name")
parser.add_argument('--outlayer', type=str, default='Linear', help="Output aggregation type")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Hidden dimension")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Time embedding dimension")
parser.add_argument('-nd', '--node_dim', type=int, default=10, help="Node embedding dimension")
parser.add_argument('--gpu', type=str, default='0', help='GPU id')


args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()

print("torch.cuda.is_available()", torch.cuda.is_available())
print("PID, device:", args.PID, args.device)


#####################################################################################################

if __name__ == '__main__':
    utils.setup_seed(args.seed)

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)

    ##################################################################
    data_obj = parse_datasets(args, patch_ts=True)
    input_dim = data_obj["input_dim"]

    # Model setting
    args.ndim = input_dim
    model = tPatchGNN(args).to(args.device)

    ##################################################################
    if args.n < 12000:
        args.state = "debug"
        log_path = f"logs/{args.dataset}_{args.model}_{args.state}.log"
    else:
        log_path = ("logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr.log"
                    .format(args.dataset, args.model, args.state,
                            args.patch_size, args.stride, args.nlayer, args.lr))

    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path,
                              filepath=os.path.abspath(__file__),
                              mode=args.logmode)

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_batches = data_obj["n_train_batches"]
    print("n_train_batches:", num_batches)

    best_val_mse = np.inf
    test_res = None
    for itr in range(args.epoch):
        st = time.time()

        ### Training ###
        model.train()
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            train_res = compute_all_losses(model, batch_dict)
            train_res["loss"].backward()
            optimizer.step()


        ### Validation ###
        model.eval()
        with torch.no_grad():
            val_res = evaluation(model, data_obj["val_dataloader"],
                                 data_obj["n_val_batches"])

            ### Testing ###
            if val_res["mse"] < best_val_mse:
                best_val_mse = val_res["mse"]
                best_iter = itr
                test_res = evaluation(model, data_obj["test_dataloader"],
                                      data_obj["n_test_batches"])

            logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
            logger.info("Train - Loss (one batch): {:.5f}"
                        .format(train_res["loss"].item()))
            logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%"
                        .format(val_res["loss"], val_res["mse"],
                                val_res["rmse"], val_res["mae"],
                                val_res["mape"] * 100))
            if test_res is not None:
                logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%"
                            .format(best_iter, test_res["loss"], test_res["mse"],
                                    test_res["rmse"], test_res["mae"], test_res["mape"] * 100))
            logger.info("Time spent: {:.2f}s".format(time.time() - st))

        if (itr - best_iter >= args.patience):
            print("Exp has been early stopped!")
            sys.exit(0)
