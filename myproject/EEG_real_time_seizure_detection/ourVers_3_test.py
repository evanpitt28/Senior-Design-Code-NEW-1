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

import os
import numpy as np
import random
import time
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn

#from builder.data.data_preprocess import get_data_preprocessed
from builder.models.detector_models.commented_resnet_lstm import CNN2D_LSTM_V8_4  # Import ResNetLSTM model directly (external file needed)
from builder.utils.metrics import Evaluator
from builder.utils.logger import Logger
# Instead of importing from builder.trainer, use the correct module
from builder.trainer.OurVers_trainer import sliding_window_v2
from builder.utils.utils import set_seeds, set_devices

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
    device = 0 # GPU device number to use
    binary_target_groups = 2
    output_dim = 2
    data_path = '/path/to/data_directory/data_path'
    dir_root = os.getcwd()
    dir_result = '/path/to/results_directory'
    num_layers = 2
    dropout = 0.1
    num_channel =  32 # Number of data channels (e.g., EEG channels)
    sincnet_bandnum = 20 # SincNet configuration
    enc_model = "sincnet"
    window_size = 1
    window_size_sig = 1 #added 12/2
    sincnet_kernel_size = 81
    sincnet_layer_num = 1
    cnn_channel_sizes = [20, 10, 10]
    sincnet_stride = 2
    sincnet_input_normalize = "none"
    window_shift_label = 1
    window_size_label = 1
    requirement_target = None
    feature_sample_rate = 256
    ignore_model_speed = False

args = Args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# test_mode
label_method_max = True
scheduler = None
optimizer = None
criterion = nn.CrossEntropyLoss(reduction='none')
iteration = 1
set_seeds(args)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#logger = Logger(args)
logger = 0
print("Project name is: ", args.project_name)

# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
#print("args: ", args)

# Dummy get_data_preprocessed function for testing purposes
def get_data_preprocessed(args):
    train_loader = [([], [], [], [], [], [])]  # Placeholder empty values
    val_loader = [([], [], [], [], [], [])]  # Placeholder empty values
    test_loader = [([], [], [], [], [], [])]  # Placeholder empty values
    len_train_dir = 0
    len_val_dir = 0
    len_test_dir = 0
    return train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir

# Get Dataloader, Model
train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocessed(args)


model = CNN2D_LSTM_V8_4(args, device).to(device)  # Directly initialize ResNetLSTM and move to the appropriate device (CPU, GPU, or MPS)
evaluator = Evaluator(args)
names = [args.project_name]
average_speed_over = 10
time_taken = 0
num_windows = 30 - args.window_size

for name in names: 
    # Check if checkpoint exists
    if args.last:
        ckpt_path = args.dir_result + '/' + name + '/ckpts/best_{}.pth'.format(str(args.seed))
    elif args.best:
        ckpt_path = args.dir_result + '/' + name + '/ckpts/best_{}.pth'.format(str(args.seed))

    #args.data_path = ''
    #args.dir_root = os.getcwd()
    #args.dir_result = '/Users/evanpittman/Downloads/Purdue Y1/Important Docs/Senior Year/Senior Design Code NEW/Iterations_HERE/Senior-Design-Code-NEW-1-6/myproject' 
    #ckpt_path = '/Users/evanpittman/Downloads/Purdue Y1/Important Docs/Senior Year/Senior Design Code NEW/Iterations_HERE/Senior-Design-Code-NEW-1-6/myproject/best_model_weights.pth'
    ckpt_path = '/Users/aprib/Downloads/best_model_weights.pth'
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    #print(ckpt.keys())

    #state = {k: v for k, v in ckpt['model'].items()}
    model.load_state_dict(ckpt, strict=False)
    
    model.eval()
    print('loaded model')
    print("Test type is: ", args.test_type)
    evaluator.reset()
    result_list = []
    iteration = 0
    evaluator.seizure_wise_eval_for_binary = True


    with torch.no_grad():
        for test_batch in tqdm(test_loader, total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            test_x, test_y, seq_lengths, target_lengths, aug_list, signal_name_list = test_batch
            #test_x = test_x.to(device)
            iteration += 1
            ### Model Structures
            print(f'iteration : {iteration}')
            iteration_start = time.time()
            if args.task_type == "binary": 
                model, _ = sliding_window_v2(args, iteration, test_x, test_y, seq_lengths, 
                                            target_lengths, model, logger, device, scheduler,
                                            optimizer, criterion, signal_name_list=signal_name_list, flow_type="test")
            else:
                print("Selected trainer is not prepared yet...")
                exit(1)
            
            if not args.ignore_model_speed:
                iteration_end = time.time()
                print("Number of windows: ", num_windows)
                print("Used device: ", device)
                print("Number of cpu threads: {}".format(torch.get_num_threads()))

                print(f'Time taken to iterate once :    {(iteration_end-iteration_start)} seconds')
                print(f'Time taken per window slide :    {(iteration_end-iteration_start)/num_windows} seconds')
                exit(1)

    #logger.test_result_only()