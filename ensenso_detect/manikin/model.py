#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from random import shuffle

torch.set_default_tensor_type('torch.DoubleTensor')

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
class LSTMModel(nn.Module):
    '''
    nn.LSTM Parameters:
        input_size  – The number of expected features in the input x
        hidden_size – The number of features in the hidden state h
        num_layers  – Number of recurrent layers.
    Inputs: input, (h_0, c_0)
        input (seq_len, batch, input_size)
        h_0 (num_layers * num_directions, batch, hidden_size)
        c_0 (num_layers * num_directions, batch, hidden_size)
    Outputs: output, (h_n, c_n)
        output (seq_len, batch, hidden_size * num_directions)
        h_n (num_layers * num_directions, batch, hidden_size)
        c_n (num_layers * num_directions, batch, hidden_size)
    '''
    def __init__(self, inputSize=64, nHidden=32, batchSize=1):
        super(LSTMModel, self).__init__()

        self.input_size  = inputSize
        self.hidden_size = nHidden
        self.batch_size  = batchSize

class StackRegressive(nn.Module)
    def __init__(self, inputSize=64, nHidden=[64,32,16], noutputs=12, batchSize=1, \
                 numLayers=2):
        super.__init__(StackRegressive, self):
        """
        Following the interpretable learning from self-driving examples:
        https://arxiv.org/pdf/1703.10631.pdf   we can extract the last
        feature cube x_t from the resnet model as a set of L = W x H
        vectors of depth D.

        Since these share the same feature extraction layers, only the
        final regression layers need to be recomputed after computing the
        classification network

        We then stack an LSTM module on this layer to obtain the detection
        predictions

        The number of outputs is thus given:
        First 4 cols represent top and lower coordinates of face boxes,
        Followed by 2 cols belonging to left eye pixel coords,
        last 2 cols are the right eye coords
        """
        def build_regressor(self, model='resnet_acc=97_iter=1000.pkl'):
            #obtain model
            res_classifier = ResNet(ResidualBlock, [3, 3, 3])
    		res_classifier.load_state_dict(torch.load('models225/' + model))

            # Get everything but the last last_layer
            res_feature_cube = nn.Sequential(*list(res_classifier.children())[:-1])

            return res_feature_cube
        self.res_classifier = build_regressor

        # Backprop Through Time (Recurrent Layer) Params
        self.cost = nn.MSELoss(size_average=False)
        self.noutputs = noutputs
        self.num_layers = numLayers
        self.input_size  = inputSize
        self.hidden_size = nHidden
        self.batch_size  = batchSize
        self.noutputs = noutputs
        self.criterion = nn.MSELoss(size_average=False)

        """
        Now stack an LSTM on top of the convnet to generate bounding box predictions
        Since last conv layer in classifier is a 64-layer, we initiate our LSTM with
        a 64-neuron input layer
        """

        #define the recurrent connections
        self.lstm1 = nn.LSTM(inputSize, nHidden[0], numLayers)
        self.lstm2 = nn.LSTM(nHidden[0], nHidden[1], numLayers)
        self.lstm3 = nn.LSTM(nHidden[1], nHidden[2], noutputs)
        self.fc    = nn.Linear(nHidden[2], noutputs)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        nBatch = x.size(0)

        out = self.build_regressor(x)
        #set initial states
        h0 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[0]).cuda())
        c0 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[0]).cuda())
        # Forward propagate RNN layer 1
        out, _ = self.lstm1(x, (h0, c0))
        out = self.drop(out)

        # Set hidden layer 2 states
        h1 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[1]).cuda())
        c1 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[1]).cuda())
        # Forward propagate RNN layer 2
        out, _ = self.lstm2(out, (h1, c1))

        # Set hidden layer 3 states
        h2 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[2]).cuda())
        c2 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[2]).cuda())
        # Forward propagate RNN layer 2
        out, _ = self.lstm3(out, (h2, c2))
        out = self.drop(out)

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])

        out = out.view(nBatch, -1)
