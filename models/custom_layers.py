import torch
import torch.nn as nn

class CustomConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        out = self.conv(x)
        return torch.sigmoid(out) * out

class AttentionLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attn(x)
        return x * attn

class CustomActivation(nn.Module):
    def forward(self, x):
        return x * torch.tanh(x)

class CustomPooling(nn.Module):
    def forward(self, x):
        return torch.max_pool2d(x, kernel_size=2)
