import torch
from torch import nn, optim
from torch.nn import functional as F
from FCN import FullyConvNet

class LMCNet(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(LMCNet, self).__init__()
        self.fcn = FullyConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.fc1 = nn.Linear(15488, 1024)
        # self.fc1_bn =nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)

        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)

    def forward(self, x):
        x=self.fcn(x)
        x=torch.flatten(x,start_dim=1)
        x=F.sigmoid(self.fc1(x))
        x=F.softmax(self.fc2(x))

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class MCNet(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(MCNet, self).__init__()
        self.fcn = FullyConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.fc1 = nn.Linear(15488, 1024)
        # self.fc1_bn =nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)

        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)

    def forward(self, x):
        x=self.fcn(x)
        x=torch.flatten(x,start_dim=1)
        x=F.sigmoid(self.fc1(x))
        x=F.softmax(self.fc2(x))

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class MLMCNet(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(MLMCNet, self).__init__()
        self.fcn = FullyConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.fc1 = nn.Linear(49728, 1024)
        # self.fc1_bn =nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)

        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)

    def forward(self, x):
        x=self.fcn(x)
        x=torch.flatten(x,start_dim=1)
        x=F.sigmoid(self.fc1(x))
        x=F.softmax(self.fc2(x))

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

# WORK IN PROGRESS
# class TSCNN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
#         super(TSCNN, self).__init__()
#         self.fcn = FullyConvNet(input_size, num_channels, kernel_size, dropout=dropout)
#
#         self.fc1 = nn.Linear(15488, 1024)
#         # self.fc1_bn =nn.BatchNorm1d(1024)
#         self.fc2 = nn.Linear(1024, 10)
#
#         self.initialise_layer(self.fc1)
#         self.initialise_layer(self.fc2)
#
#     def forward(self, x):
#         x=self.fcn(x)
#         x=torch.flatten(x,start_dim=1)
#         x=F.sigmoid(self.fc1(x))
#         x=F.softmax(self.fc2(x))
#
#         return x
