import torch
from torch import nn, optim
from torch.nn import functional as F
from FCN import FullyConvNet

class LMCNet(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super(LMCNet, self).__init__()

        self.dropout=nn.Dropout(p=dropout)

        self.fcn = FullyConvNet(height, width, channels, class_count, dropout=dropout)

        self.fc1 = nn.Linear(15488, 1024)
        # self.fc1_bn =nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)

        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.fcn(x)
        x=torch.flatten(x,start_dim=1)
        x=F.sigmoid(self.fc1(self.dropout(x)))
        x=self.fc2(x)

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class MCNet(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super(MCNet, self).__init__()

        self.dropout=nn.Dropout(p=dropout)

        self.fcn = FullyConvNet(height, width, channels, class_count, dropout=dropout)

        self.fc1 = nn.Linear(15488, 1024)
        # self.fc1_bn =nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)

        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.fcn(x)
        x=torch.flatten(x,start_dim=1)
        x=F.sigmoid(self.fc1(self.dropout(x)))
        x=self.fc2(x)

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class MLMCNet(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super(MLMCNet, self).__init__()

        self.dropout=nn.Dropout(p=dropout)

        self.fcn = FullyConvNet(height, width, channels, class_count, dropout=dropout)

        self.fc1 = nn.Linear(26048, 1024)
        # self.fc1_bn =nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)

        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.fcn(x)
        x=torch.flatten(x,start_dim=1)
        x=F.sigmoid(self.fc1(self.dropout(x)))
        x=self.fc2(x)

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
