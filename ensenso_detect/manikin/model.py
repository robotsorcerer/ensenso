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
    def __init__(self, block, layers, num_classes=2):
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
        # print(out.size())
        out = self.fc(out)
        return out

class StackRegressive(nn.Module):
    def __init__(self, **kwargs):
        super(StackRegressive, self).__init__()
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

        self.criterion = nn.MSELoss(size_average=False)
        # Backprop Through Time (Recurrent Layer) Params
        self.model          = kwargs['model']
        self.noutputs       = kwargs['noutputs']
        self.num_layers     = kwargs['numLayers']
        self.input_size     = kwargs['inputSize']
        self.hidden_size    = kwargs['nHidden']
        self.batch_size     = kwargs['batchSize']
        self.noutputs       = kwargs['noutputs']
        self.cuda           = kwargs['cuda']
        self.res_cube       = kwargs['res_classifier']

        for key in kwargs:
            # print("(%s: %s)" % (key, kwargs[key]))
            self.key = kwargs[key]
            # print(type(key))

        self.criterion = nn.MSELoss(size_average=False)
        self.fc = nn.Linear(64, self.noutputs)

        #obtain model
        def build_regressor(self):
            #extract feture cube of last layer and reshape it
            res_classifier = ResNet(ResidualBlock, [3, 3, 3])

            if args.classifier is not None:    #use pre-trained classifier
        	      res_classifier.load_state_dict(torch.load('models225/' + args.classifier))

            # Get everything but the classifier fc (last) layer
            res_cube = list(res_classifier.children()).pop()
            #reshape last layer for input of bounding box coords
            res_cube.append(nn.Linear(inputSize), inputSize)
            return res_feature_cube
        self.res_classifier = build_regressor

        """
        Now stack an LSTM on top of the convnet to generate bounding box predictions
        Since last conv layer in classifier is a 64-layer, we initiate our LSTM with
        a 64-neuron input layer
        """

        #define the recurrent connections
        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size[0], self.num_layers)
        self.lstm2 = nn.LSTM(self.hidden_size[0], self.hidden_size[1], self.num_layers)
        self.lstm3 = nn.LSTM(self.hidden_size[1], self.hidden_size[2], self.noutputs)
        self.fc    = nn.Linear(self.hidden_size[2], self.noutputs)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        nBatch = x.size(0)

        # out = self.res_classifier(x)

        #set initial states
        h0 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[0]))
        c0 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[0]))

        if self.cuda:
            h0.data = h0.data.cuda()
            c0.data = c0.data.cuda()
        # Forward propagate RNN layer 1
        out, _ = self.lstm1(x, (h0, c0))
        out = self.drop(out)

        # Set hidden layer 2 states
        h1 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[1]))
        c1 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[1]))

        if self.cuda:
            h1.data = h1.data.cuda()
            c1.data = c1.data.cuda()
        # Forward propagate RNN layer 2
        out, _ = self.lstm2(out, (h1, c1))

        # Set hidden layer 3 states
        h2 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[2]))
        c2 = Variable(torch.Tensor(self.num_layers, self.batch_size, self.hidden_size[2]))

        if self.cuda:
            h2.data = h2.data.cuda()
            c2.data = c2.data.cuda()

        # Forward propagate RNN layer 2
        out, _ = self.lstm3(out, (h2, c2))
        out = self.drop(out)

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])

        out = out.view(nBatch, -1)

        return out
