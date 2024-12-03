# Copyright (c) 2022, Kwanhyung Lee, Hyewon Jeong, Seyun Kim AITRICS. All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#this calls get_deta_preprocessed from data_preprocess.py, which is a function that returns the train, validation, and test data loaders AND works with the eeg_binary_collate_fn(train_data) function which is in charge of the real-time data augmentation, soooooo we still need to mess around with that

#this also calls get_detector_model from models.py, which is a function that returns the model based on the configuration arguments, and the goal is to ignore that and rewrite it to use the resnet_lstm model

#big goal is to have one python file that holds all the important info from data_preprocess.py, 1_preprocess.py, resnet_lstm.py, and this file 2_train.py, (plus all the necessary builder and control files) and be able to run it all in one go

# Importing required libraries and modules
import numpy as np
import os
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from itertools import groupby
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
#from torchinfo import summary

# Custom modules and utilities (required external Python files)
from builder.utils.lars import LARC  # Optimizer utility (external file needed)
from builder.data.data_preprocess import get_data_preprocessed  # Data preprocessing function (external file needed)
#from builder.models import get_detector_model, grad_cam  # Model definition and Grad-CAM utility (external files needed)
from builder.models.detector_models.commented_resnet_lstm import CNN2D_LSTM_V8_4  # Import ResNetLSTM model directly (external file needed)
from builder.utils.logger import Logger  # Logger utility to track training and validation metrics (external file needed)
from builder.utils.utils import set_seeds, set_devices  # Utility functions for setting seeds and devices (external file needed)
from builder.utils.cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts  # Custom learning rate scheduler (external file needed)
from builder.utils.cosine_annealing_with_warmupSingle import CosineAnnealingWarmUpSingle  # Custom learning rate scheduler (external file needed)
from builder.trainer import get_trainer  # Trainer utility for training steps (external file needed)
from builder.trainer import *  # Additional trainer utilities (external file needed)

# Setting CUDA device order to ensure consistent GPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# List to store test results for each seed
list_of_test_results_per_seed = []

# Define result class

# Manually set arguments here instead of using argparse
class Args:
    seed_list = [0, 1004, 911, 2021, 119]
    seed = 10
    project_name = "test_project"
    checkpoint = False
    epochs = 10
    batch_size = 32
    optim = 'adam'
    lr_scheduler = "Single"
    lr_init = 1e-3
    lr_max = 4e-3
    t_0 = 5
    t_mult = 2
    t_up = 1
    gamma = 0.5
    momentum = 0.9
    weight_decay = 1e-6
    task_type = 'binary'
    log_iter = 10
    best = True
    last = False
    test_type = "test"
    device = 0  # GPU device number to use

    binary_target_groups = 2
    output_dim = 2

    #added by alyssa
    #dir_result = path_configs['dir_result']

    """    # target groups options
    # "1": '0':'bckg', '1':'gnsz', '2':'fnsz', '3':'spsz', '4':'cpsz', '5':'absz', '6':'tnsz', '7':'tcsz', '8':'mysz'
    # "2": '0':'bckg', '1':'gnsz_fnsz_spsz_cpsz_absz_tnsz_tcsz_mysz'
    # "4": '0':'bckg', '1':'gnsz_absz', '2':'fnsz_spsz_cpsz', '3':'tnsz', '4':'tcsz', '5':'mysz'
    parser.add_argument('--binary-target-groups', type=int, default=2, choices= [1, 2, 3])
    parser.add_argument('--eeg-type', type=str, default="bipolar", choices=["unipolar", "bipolar", "uni_bipolar"])
    parser.add_argument('--task-type', '-tt', type=str, default='binary', choices=['anomaly', 'multiclassification', 'binary', 'binary_noslice'])    """


def initialize_model(args, device):
    # Create the model
    model = CNN2D_LSTM_V8_4(args).to(device)  # Directly initialize ResNetLSTM and move to the appropriate device (CPU, GPU, or MPS)
    return model

def load_checkpoint(args, model, device, logger, seed_num):
    # Load checkpoint if specified
    if args.checkpoint:
        if args.last:
            ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/best_{}.pth'.format(str(seed_num))
        elif args.best:
            ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/best_{}.pth'.format(str(seed_num))
        checkpoint = torch.load(ckpt_path, map_location=device)  # Load model checkpoint from file
        model.load_state_dict(checkpoint['model'])  # Load saved model state
        logger.best_auc = checkpoint['score']  # Set best AUC score from checkpoint
        start_epoch = checkpoint['epoch']  # Set starting epoch from checkpoint
        del checkpoint
    else:
        logger.best_auc = 0
        start_epoch = 1
    return model, start_epoch

def set_optimizer(args, model):
    # Set up the optimizer based on specified argument
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    elif args.optim == 'adam_lars':
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)  # LARS wrapper for adaptive learning rate scaling
    elif args.optim == 'sgd_lars':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
    elif args.optim == 'adamw_lars':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
        optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
    return optimizer

def set_scheduler(args, optimizer, one_epoch_iter_num):
    # Set up learning rate scheduler
    if args.lr_scheduler == "CosineAnnealing":
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.t_0 * one_epoch_iter_num, T_mult=args.t_mult, eta_max=args.lr_max, T_up=args.t_up * one_epoch_iter_num, gamma=args.gamma)  # Custom cosine annealing scheduler with warmup (external file needed)
    elif args.lr_scheduler == "Single":
        scheduler = CosineAnnealingWarmUpSingle(optimizer, max_lr=args.lr_init * math.sqrt(args.batch_size), epochs=args.epochs, steps_per_epoch=one_epoch_iter_num, div_factor=math.sqrt(args.batch_size))  # Alternative scheduler (external file needed)
    return scheduler

def train_model(args, model, train_loader, val_loader, device, logger, optimizer, scheduler, start_epoch):
    # Set model to training mode
    model.train()
    iteration = 0
    logger.loss = 0

    # Start training process
    start = time.time()  # Start time for tracking training duration
    pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")  # Progress bar for tracking training progress
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_losses = []
        loss = 0

        # Iterate through each batch of training data
        for train_batch in train_loader:
            train_x, train_y, seq_lengths, target_lengths, aug_list, signal_name_list = train_batch  # Unpack training batch
            train_x, train_y = train_x.to(device), train_y.to(device)  # Move data to appropriate device (CPU, GPU, or MPS)
            iteration += 1

            # Train the model and get loss
            model, iter_loss = get_trainer(args, iteration, train_x, train_y, seq_lengths, target_lengths, model, logger, device, scheduler, optimizer, nn.CrossEntropyLoss(reduction='none'), signal_name_list)  # Perform a training step (external file needed)
            logger.loss += np.mean(iter_loss)

            # Logging training progress
            if iteration % args.log_iter == 0:
                logger.log_tqdm(epoch, iteration, pbar)  # Log progress in tqdm
                logger.log_scalars(iteration)  # Log scalar metrics (e.g., loss)

            # Validation at intervals during training
            if iteration % (len(train_loader) // 10) == 0:
                validate_model(args, model, val_loader, device, logger, iteration, scheduler, optimizer)
        pbar.update(1)
    return model

def validate_model(args, model, val_loader, device, logger, iteration, scheduler, optimizer):
    model.eval()  # Set model to evaluation mode
    logger.evaluator.reset()  # Reset evaluator for validation
    val_iteration = 0
    logger.val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            val_x, val_y, seq_lengths, target_lengths, aug_list, signal_name_list = batch  # Unpack validation batch
            val_x, val_y = val_x.to(device), val_y.to(device)  # Move data to appropriate device (CPU, GPU, or MPS)
            model, val_loss = get_trainer(args, iteration, val_x, val_y, seq_lengths, target_lengths, model, logger, device, scheduler, optimizer, nn.CrossEntropyLoss(reduction='none'), signal_name_list, flow_type=args.test_type)  # Perform validation step
            logger.val_loss += np.mean(val_loss)
            val_iteration += 1
        
        logger.log_val_loss(val_iteration, iteration)  # Log validation loss
        logger.add_validation_logs(iteration)  # Add validation metrics to log
        logger.save(model, optimizer, iteration, iteration)  # Save model checkpoint
    model.train()  # Set model back to training mode

def main():
    args = Args()

    # Set the device (MPS for Apple silicon, CUDA for Nvidia GPUs, or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define result classes for validation and test results
    #save_valid_results = experiment_results_validation(args)
    #save_test_results = experiment_results(args)

    # Loop through each seed to train and evaluate the model
    for seed_num in args.seed_list:
        # Set the seed for reproducibility
        args.seed = seed_num
        set_seeds(args)

        # Initialize the logger to track training and validation metrics
        logger = Logger(args)  # Logger instance to save metrics
        logger.evaluator.best_auc = 0

        # Load preprocessed data (train, validation, test)
        train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocessed(args)  # Data loaders for training, validation, and testing

        # Initialize model
        model = initialize_model(args, device)

        # Load checkpoint if available
        model, start_epoch = load_checkpoint(args, model, device, logger, seed_num)

        # Set up optimizer and scheduler
        optimizer = set_optimizer(args, model)
        one_epoch_iter_num = len(train_loader)  # Total number of iterations per epoch
        scheduler = set_scheduler(args, optimizer, one_epoch_iter_num)

        # Train model
        model = train_model(args, model, train_loader, val_loader, device, logger, optimizer, scheduler, start_epoch)

        # Log validation results
        logger.val_result_only()  # Log only validation results
        save_valid_results.results_all_seeds(logger.test_results)  # Save validation results

if __name__ == "__main__":
    main()
