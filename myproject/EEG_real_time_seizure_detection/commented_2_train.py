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

# Importing required libraries and modules
import numpy as np
import os
import argparse
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
from torchinfo import summary

# Custom modules and utilities (required external Python files)
from builder.utils.lars import LARC  # Optimizer utility (external file needed)
from control.config import args  # Configuration arguments (external file needed)
from builder.data.data_preprocess import get_data_preprocessed  # Data preprocessing function (external file needed)
from builder.models import get_detector_model, grad_cam  # Model definition and Grad-CAM utility (external files needed)
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

# Define result classes for validation and test results
save_valid_results = experiment_results_validation(args)  # Handles storing validation results (external file needed)
save_test_results = experiment_results(args)  # Handles storing test results (external file needed)

# Loop through each seed to train and evaluate the model
for seed_num in args.seed_list:
    # Set the seed for reproducibility
    args.seed = seed_num
    set_seeds(args)
    
    # Set the device (CPU or GPU) to be used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Initialize the logger to track training and validation metrics
    logger = Logger(args)  # Logger instance to save metrics
    logger.evaluator.best_auc = 0

    # Load preprocessed data (train, validation, test)
    train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocessed(args)  # Data loaders for training, validation, and testing
    
    # Create the model
    model = get_detector_model(args)  # Initialize model based on configuration arguments
    val_per_epochs = 10  # Number of times validation is performed per epoch
    model = model(args, device).to(device)  # Move model to appropriate device (CPU or GPU)
    
    # Define the loss criterion (CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Loss function used for classification tasks

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

    # Calculate the number of iterations per epoch
    one_epoch_iter_num = len(train_loader)  # Total number of iterations per epoch
    print("Iterations per epoch: ", one_epoch_iter_num)
    iteration_num = args.epochs * one_epoch_iter_num  # Total number of iterations for all epochs

    # Set up learning rate scheduler
    if args.lr_scheduler == "CosineAnnealing":
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.t_0*one_epoch_iter_num, T_mult=args.t_mult, eta_max=args.lr_max, T_up=args.t_up*one_epoch_iter_num, gamma=args.gamma)  # Custom cosine annealing scheduler with warmup (external file needed)
    elif args.lr_scheduler == "Single":
        scheduler = CosineAnnealingWarmUpSingle(optimizer, max_lr=args.lr_init * math.sqrt(args.batch_size), epochs=args.epochs, steps_per_epoch=one_epoch_iter_num, div_factor=math.sqrt(args.batch_size))  # Alternative scheduler (external file needed)

    # Set model to training mode
    model.train()
    iteration = 0
    logger.loss = 0

    # Start training process
    start = time.time()  # Start time for tracking training duration
    pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")  # Progress bar for tracking training progress
    for epoch in range(start_epoch, args.epochs+1):
        epoch_losses = []
        loss = 0

        # Iterate through each batch of training data
        for train_batch in train_loader:
            train_x, train_y, seq_lengths, target_lengths, aug_list, signal_name_list = train_batch  # Unpack training batch
            train_x, train_y = train_x.to(device), train_y.to(device)  # Move data to appropriate device (CPU or GPU)
            iteration += 1
         
            # Train the model and get loss
            model, iter_loss = get_trainer(args, iteration, train_x, train_y, seq_lengths, target_lengths, model, logger, device, scheduler, optimizer, criterion, signal_name_list)  # Perform a training step (external file needed)
            logger.loss += np.mean(iter_loss)

            # Logging training progress
            if iteration % args.log_iter == 0:
                logger.log_tqdm(epoch, iteration, pbar)  # Log progress in tqdm
                logger.log_scalars(iteration)  # Log scalar metrics (e.g., loss)

            # Validation at intervals during training
            if iteration % (one_epoch_iter_num//val_per_epochs) == 0:
                model.eval()  # Set model to evaluation mode
                logger.evaluator.reset()  # Reset evaluator for validation
                val_iteration = 0
                logger.val_loss = 0
                with torch.no_grad():
                    for idx, batch in enumerate(tqdm(val_loader)):
                        val_x, val_y, seq_lengths, target_lengths, aug_list, signal_name_list = batch  # Unpack validation batch
                        val_x, val_y = val_x.to(device), val_y.to(device)  # Move data to appropriate device
                        model, val_loss = get_trainer(args, iteration, val_x, val_y, seq_lengths, target_lengths, model, logger, device, scheduler, optimizer, criterion, signal_name_list, flow_type=args.test_type)  # Perform validation step (external file needed)
                        logger.val_loss += np.mean(val_loss)
                        val_iteration += 1
                    
                    logger.log_val_loss(val_iteration, iteration)  # Log validation loss
                    logger.add_validation_logs(iteration)  # Add validation metrics to log
                    logger.save(model, optimizer, iteration, epoch)  # Save model checkpoint
                model.train()  # Set model back to training mode
        pbar.update(1)

    # Log validation results
    logger.val_result_only()  # Log only validation results
    save_valid_results.results_all_seeds(logger.test_results)  # Save validation results
    
    # Reset model and load best model checkpoint for testing
    del model
    model = get_detector_model(args)  # Reinitialize the model
    val_per_epochs = 2

    print("#################################################")
    print("################# Test Begins ###################")
    print("#################################################")
    model = model(args, device).to(device)  # Move model to appropriate device
    logger = Logger(args)  # Initialize a new logger for testing
    # Load model checkpoint for testing
    if args.last:
        ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/last.pth'
    elif args.best:
        ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/best.pth'

    if not os.path.exists(ckpt_path):
        print("Final model for test experiment doesn't exist...")
        exit(1)
    # Load model & state
    ckpt = torch.load(ckpt_path, map_location=device)  # Load checkpoint
    state = {k: v for k, v in ckpt['model'].items()}  # Extract model state
    model.load_state_dict(state)  # Load state into model

    # Initialize test step
    model.eval()  # Set model to evaluation mode
    logger.evaluator.reset()  # Reset evaluator for testing
    
    with torch.no_grad():
        for test_batch in tqdm(test_loader, total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            test_x, test_y, seq_lengths, target_lengths, aug_list, signal_name_list = test_batch  # Unpack test batch
            test_x, test_y = test_x.to(device), test_y.to(device)  # Move data to appropriate device
            
            # Perform testing and log results
            model, _ = get_trainer(args, iteration, test_x, test_y, seq_lengths, 
                                        target_lengths, model, logger, device, scheduler,
                                        optimizer, criterion, signal_name_list=signal_name_list, flow_type="test")  # Perform test step (external file needed)

    # Log test results
    logger.test_result_only()  # Log only test results
    list_of_test_results_per_seed.append(logger.test_results)  # Append test results for current seed
    logger.writer.close()  # Close logger writer

# Collect and print final test results for each seed
auc_list = []
apr_list = []
f1_list = []
tpr_list = []
tnr_list = []
os.system("echo  '#######################################'")
os.system("echo  '##### Final test results per seed #####'")
os.system("echo  '#######################################'")
for result, tpr, tnr in list_of_test_results_per_seed:    
    os.system("echo  'seed_case:{} -- auc: {}, apr: {}, f1_score: {}, tpr: {}, tnr: {}'".format(str(result[0]), str(result[1]), str(result[2]), str(result[3]), str(tpr), str(tnr)))
    auc_list.append(result[1])
    apr_list.append(result[2])
    f1_list.append(result[3])
    tpr_list.append(tpr)
    tnr_list.append(tnr)
os.system("echo  'Total average -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}'".format(str(np.mean(auc_list)), str(np.mean(apr_list)), str(np.mean(f1_list)), str(np.mean(tpr_list)), str(np.mean(tnr_list))))
os.system("echo  'Total std -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}'".format(str(np.std(auc_list)), str(np.std(apr_list)), str(np.std(f1_list)), str(np.std(tpr_list)), str(np.std(tnr_list))))
