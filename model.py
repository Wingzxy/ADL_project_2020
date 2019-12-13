
# THIS IS THE EXACT CNN FROM LAB 4. WE WILL MODIFY THIS



import torch
from torch import nn, optim

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.dropout=nn.Dropout(p=dropout)

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5,5),
            padding=(2,2),
        )
        self.conv2_bn =nn.BatchNorm2d(64)

        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        ## TASK 2-1: Define the second convolutional layer and initialise its parameters
        self.initialise_layer(self.conv2)
        ## TASK 3-1: Define the second pooling layer
        ## TASK 5-1: Define the first FC layer and initialise its parameters
        self.fc1 = nn.Linear(4096, 1024)
        self.fc1_bn =nn.BatchNorm1d(1024)
        self.initialise_layer(self.fc1)
        ## TASK 6-1: Define the last FC layer and initialise its parameters
        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1_bn(self.conv1(images)))
        x = self.pool1(x)
        ## TASK 2-2: Pass x through the second convolutional layer
        x=F.relu(self.conv2_bn(self.conv2(x)))
        ## TASK 3-2: Pass x through the second pooling layer
        x=self.pool1(x)
        ## TASK 4: Flatten the output of the pooling layer so it is of shape
        ##         (batch_size, 4096)
        x=torch.flatten(x,start_dim=1)
        ## TASK 5-2: Pass x through the first fully connected layer
        x=F.relu(self.fc1_bn(self.fc1(x)))
        ## TASK 6-2: Pass x through the last fully connected layer
        x=self.fc2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
