import torch
from torch import nn, optim
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.dropout=nn.Dropout(p=dropout)

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
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

        self.fc1 = nn.Linear(15488, 1024)
        # self.fc1_bn =nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)

        self.initialise_layer(self.conv1)
        self.initialise_layer(self.conv2)
        self.initialise_layer(self.conv3)
        self.initialise_layer(self.conv4)

        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x=F.relu(self.conv1_bn(self.conv1(images)))
        x=F.relu(self.conv2_bn(self.conv2(x)))
        x=self.pool(x)
        x=F.relu(self.conv3_bn(self.conv3(x)))
        x=F.relu(self.conv4_bn(self.conv4(x)))

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
