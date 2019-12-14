import torch
from torch import nn, optim
from torch.nn import functional as F


class FullyConvNet(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.class_count = class_count

        self.dropout=nn.Dropout(p=dropout)

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.conv2_bn = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.conv3_bn =nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(2, 2),
        )
        self.conv4_bn =nn.BatchNorm2d(64)

        self.initialise_layer(self.conv1)
        self.initialise_layer(self.conv2)
        self.initialise_layer(self.conv3)
        self.initialise_layer(self.conv4)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=F.relu(self.conv1_bn(self.conv1(x)))
        x=F.relu(self.conv2_bn(self.conv2(x)))
        x=self.pool(x)
        x=F.relu(self.conv3_bn(self.conv3(x)))
        x=F.relu(self.conv4_bn(self.conv4(x)))

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
