# Copyright (c) 2022, Kwanhyung Lee, AITRICS. All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
# Import necessary libraries
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import importlib
# Import custom feature extraction modules
from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE
from builder.models.feature_extractor.lfcc_feature import LFCC_FEATURE

# Define a basic block for ResNet. This block will be reused in the construction of ResNet layers.
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        # Input:
        # - in_planes: Number of input channels
        # - planes: Number of output channels after convolution
        # - stride: Stride used in convolution to control output size
        super(BasicBlock, self).__init__()
        
        # Define the first convolutional layer, followed by batch normalization.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Define the second convolutional layer, with stride fixed at 1, followed by batch normalization.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # If stride is greater than 1, downsample input to match the shape for addition.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # Forward pass of the BasicBlock.
        # Input: x - input tensor
        # Output: out - output tensor after applying convolutions and adding residual connection
        out = F.relu(self.bn1(self.conv1(x)))  # First conv layer, followed by batch norm and ReLU activation
        out = self.bn2(self.conv2(out))        # Second conv layer, followed by batch norm
        out += self.shortcut(x)                # Add residual (shortcut) connection to maintain information flow
        out = F.relu(out)                      # Apply ReLU activation again to introduce non-linearity
        return out

# Define a CNN2D-LSTM model for EEG signal classification
class CNN2D_LSTM_V8_4(nn.Module):
    def __init__(self, args, device):
        # Input:
        # - args: Arguments containing model configurations like number of layers, dropout, etc.
        # - device: Device where the model will be run (e.g., CPU or GPU)
        super(CNN2D_LSTM_V8_4, self).__init__()      
        self.args = args

        # Set model parameters
        self.num_layers = args.num_layers  # Number of LSTM layers
        self.hidden_dim = 256  # Number of features in the hidden state of the LSTM
        self.dropout = args.dropout  # Dropout rate for regularization
        self.num_data_channel = args.num_channel  # Number of data channels (e.g., EEG channels)
        self.sincnet_bandnum = args.sincnet_bandnum  # SincNet configuration
        self.feature_extractor = args.enc_model  # Feature extraction method

        # Initialize feature extraction model based on selected type
        if self.feature_extractor == "raw" or self.feature_extractor == "downsampled":
            pass  # No additional feature extraction needed for raw or downsampled data
        else:
            # Define feature extraction models using a dictionary
            self.feat_models = nn.ModuleDict([
                ['psd1', PSD_FEATURE1()],
                ['psd2', PSD_FEATURE2()],
                ['stft1', SPECTROGRAM_FEATURE_BINARY1()],
                ['stft2', SPECTROGRAM_FEATURE_BINARY2()],
                ['LFCC', LFCC_FEATURE()],                                
                ['sincnet', SINCNET_FEATURE(args=args, num_eeg_channel=self.num_data_channel)]  # SincNet feature extractor
            ])
            self.feat_model = self.feat_models[self.feature_extractor]  # Select the appropriate feature extractor

        # Determine the number of features for each feature extractor
        if args.enc_model == "psd1" or args.enc_model == "psd2":
            self.feature_num = 7  # PSD feature extractor outputs 7 features
        elif args.enc_model == "sincnet":
            self.feature_num = args.cnn_channel_sizes[args.sincnet_layer_num-1]  # SincNet output depends on channel size
        elif args.enc_model == "stft1":
            self.feature_num = 50  # STFT1 outputs 50 features
        elif args.enc_model == "stft2":
            self.feature_num = 100  # STFT2 outputs 100 features
        elif args.enc_model == "raw":
            self.feature_num = 1  # Raw input has only one feature channel
            self.num_data_channel = 1  # Set the number of data channels to 1 for raw input
        self.in_planes = 64  # Initial number of input planes for ResNet

        # Activation functions
        activation = 'relu'  # Use ReLU as the default activation function
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()],
            ['relu', nn.ReLU(inplace=True)],
            ['tanh', nn.Tanh()],
            ['sigmoid', nn.Sigmoid()],
            ['leaky_relu', nn.LeakyReLU(0.2)],
            ['elu', nn.ELU()]
        ])

        # Create a new variable for the hidden state, necessary to calculate the gradients
        self.hidden = (
            (torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device),
             torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device))
        )

        # Define helper functions for convolutional layers with batch normalization and activation
        def conv2d_bn(inp, oup, kernel_size, stride, padding, dilation=1):
            # Convolutional layer followed by batch normalization and activation
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
                nn.BatchNorm2d(oup),
                self.activations[activation],
            )

        def conv2d_bn_nodr(inp, oup, kernel_size, stride, padding):
            # Convolutional layer followed by batch normalization and activation (no dilation)
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(oup),
                self.activations[activation],
            )  

        # Define the first convolutional layer and pooling based on the feature extraction method
        if args.enc_model == "raw":
            self.conv1 = conv2d_bn(self.num_data_channel,  64, (1, 51), (1, 4), (0, 25))  # Raw data convolution
            self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))  # Max pooling to reduce temporal dimension
        elif args.enc_model == "sincnet":
            self.conv1 = conv2d_bn(1,  64, (7, 21), (7, 2), (0, 10))  # SincNet feature convolution
            self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))  # Max pooling
        elif args.enc_model == "psd1" or args.enc_model == "psd2" or args.enc_model == "stft2":
            self.conv1 = conv2d_bn(1,  64, (7, 21), (7, 2), (0, 10))  # PSD or STFT2 feature convolution
            self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # Max pooling
        elif args.enc_model == "LFCC":
            self.conv1 = conv2d_bn(1,  64, (8, 21), (8, 2), (0, 10))  # LFCC feature convolution
            self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # Max pooling
        elif args.enc_model == "downsampled":
            # Define multiple convolutional layers for downsampled data at different frequencies
            self.conv2d_200hz = conv2d_bn_nodr(1,  32, (1, 51), (1, 4), (0, 25))  # 200 Hz convolution
            self.conv2d_100hz = conv2d_bn_nodr(1,  16, (1, 51), (1, 2), (0, 25))  # 100 Hz convolution
            self.conv2d_50hz = conv2d_bn_nodr(1,  16, (1, 51), (1, 1), (0, 25))  # 50 Hz convolution
            self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))  # Max pooling

        # Define the ResNet layers using the BasicBlock
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)  # First ResNet layer with 64 output channels
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # Second ResNet layer with 128 output channels
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # Third ResNet layer with 256 output channels

        # Adaptive average pooling to reduce spatial dimensions to (1, 1)
        self.agvpool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM layer for temporal sequence learning
        self.lstm = nn.LSTM(
            input_size=256,  # Input size matches the output of the ResNet layers
            hidden_size=self.hidden_dim,  # Number of features in LSTM hidden state
            num_layers=args.num_layers,  # Number of LSTM layers
            batch_first=True,  # Input and output tensors are provided as (batch, seq, feature)
            dropout=args.dropout  # Dropout for regularization
        )

        # Fully connected classifier layer for outputting class probabilities
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=64, bias=True),  # Linear layer to reduce feature dimension
            nn.BatchNorm1d(64),  # Batch normalization layer
            self.activations[activation],  # Activation function
            nn.Linear(in_features=64, out_features=args.output_dim, bias=True),  # Final linear layer for classification
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        # Create a ResNet layer with multiple blocks.
        # Input:
        # - block: Block type (BasicBlock)
        # - planes: Number of output channels for this layer
        # - num_blocks: Number of blocks in this layer
        # - stride: Stride for the first block
        strides = [stride] + [1] * (num_blocks - 1)  # Set stride for the first block, others have stride of 1
        layers = []
        for stride1 in strides:
            layers.append(block(self.in_planes, planes, stride1))  # Append blocks to the layer
            self.in_planes = planes  # Update input channel size for the next block
        return nn.Sequential(*layers)  # Return the complete layer as a sequential model

    def forward(self, x):
        # Forward pass of the CNN2D-LSTM model.
        # Input: x - input tensor of shape (batch_size, channels, sequence_length)
        # Output: output - output tensor with class scores
        x = x.permute(0, 2, 1)  # Permute the input to (batch_size, channels, sequence_length)

        # Apply different feature extraction methods based on the selected type
        if self.feature_extractor == "downsampled":
            x = x.unsqueeze(1)  # Add a channel dimension
            x_200 = self.conv2d_200hz(x)  # Apply 200 Hz convolution
            x_100 = self.conv2d_100hz(x[:, :, :, ::2])  # Apply 100 Hz convolution to downsampled data
            x_50 = self.conv2d_50hz(x[:, :, :, ::4])  # Apply 50 Hz convolution to further downsampled data
            x = torch.cat((x_200, x_100, x_50), dim=1)  # Concatenate outputs along the channel dimension
            x = self.maxpool1(x)  # Apply max pooling to reduce the spatial dimensions
        elif self.feature_extractor != "raw":
            x = self.feat_model(x)  # Extract features using the selected feature extraction model
            x = x.reshape(x.size(0), -1, x.size(3)).unsqueeze(1)  # Reshape and add a channel dimension
            x = self.conv1(x)  # Apply first convolution
            x = self.maxpool1(x)  # Apply max pooling
        else:
            x = x.unsqueeze(1)  # Add a channel dimension for raw input
            x = self.conv1(x)  # Apply first convolution
            x = self.maxpool1(x)  # Apply max pooling

        # Pass through ResNet layers
        x = self.layer1(x)  # Pass through first ResNet layer
        x = self.layer2(x)  # Pass through second ResNet layer
        x = self.layer3(x)  # Pass through third ResNet layer
        x = self.agvpool(x)  # Apply adaptive average pooling to reduce spatial size to (1, 1)
        x = torch.squeeze(x, 2)  # Squeeze the height dimension to make it compatible for LSTM
        x = x.permute(0, 2, 1)  # Permute to (batch_size, sequence_length, features)

        # LSTM forward pass
        self.hidden = tuple(([Variable(var.data) for var in self.hidden]))  # Update the hidden state with current values
        output, self.hidden = self.lstm(x, self.hidden)  # Apply LSTM to learn temporal dependencies
        output = output[:, -1, :]  # Take the output from the last time step of the sequence
        output = self.classifier(output)  # Classify using the fully connected layer
        return output, self.hidden  # Output the classification results and the hidden state

    def init_state(self, device):
        # Initialize the hidden state for the LSTM
        # Input: device - The device (CPU or GPU) where the hidden state should be allocated
        # Output: Initializes the hidden state with zeros
        self.hidden = (
            (torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device),
             torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device))
        )
