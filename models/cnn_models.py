import torch.nn as nn
import torch
from models.custom_layers import CustomConvLayer, AttentionLayer, CustomActivation, CustomPooling

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.relu(x + self.conv2(self.relu(self.conv1(x))))

class CNNWithResidual(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNKernelVariants(nn.Module):
    def __init__(self, kernel_type, in_channels, num_classes):
        super().__init__()
        if kernel_type == '3x3':
            conv = nn.Conv2d(in_channels, 16, 3, padding=1)
        elif kernel_type == '5x5':
            conv = nn.Conv2d(in_channels, 16, 5, padding=2)
        elif kernel_type == '7x7':
            conv = nn.Conv2d(in_channels, 16, 7, padding=3)
        elif kernel_type == '1x1+3x3':
            conv = nn.Sequential(
                nn.Conv2d(in_channels, 8, 1),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, padding=1)
            )
        else:
            raise ValueError("Unknown kernel type")

        self.model = nn.Sequential(
            conv,
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class CNNDepthVariants(nn.Module):
    def __init__(self, depth, in_channels, num_classes):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range({'shallow': 2, 'medium': 4, 'deep': 6}.get(depth, 2)):
            layers.append(nn.Conv2d(channels, 16, 3, padding=1))
            layers.append(nn.ReLU())
            channels = 16

        if depth == 'residual':
            layers = [ResidualBlock(16) for _ in range(3)]

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNWithCustomLayers(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        if layer_type == 'conv':
            conv = CustomConvLayer(1, 16, 3)
        else:
            conv = nn.Conv2d(1, 16, 3, padding=1)

        if layer_type == 'attention':
            attn = AttentionLayer(16)
        else:
            attn = nn.Identity()

        if layer_type == 'activation':
            act = CustomActivation()
        else:
            act = nn.ReLU()

        if layer_type == 'pool':
            pool = CustomPooling()
        else:
            pool = nn.AdaptiveAvgPool2d((1, 1))

        self.model = nn.Sequential(
            conv,
            attn,
            act,
            pool
        )
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1)
        )

    def forward(self, x):
        return x + self.block(x)

class WideResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)
