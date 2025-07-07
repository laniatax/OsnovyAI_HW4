import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
from models.custom_layers import CustomConvLayer, CustomActivation, CustomPooling, AttentionLayer
from models.cnn_models import CNNWithCustomLayers
from utils.training_utils import train, evaluate
from utils.visualization_utils import plot_curves
from utils.comparison_utils import count_params
from models.cnn_models import ResidualBlock, BottleneckBlock, WideResidualBlock

logging.basicConfig(level=logging.INFO)

transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_set = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5

custom_layers = ["conv", "attention", "activation", "pool"]
for layer_type in custom_layers:
    model = CNNWithCustomLayers(layer_type=layer_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_acc, test_acc = [], []
    for epoch in range(EPOCHS):
        tr_acc, _ = train(model, train_loader, optimizer, criterion, device)
        te_acc, _ = evaluate(model, test_loader, criterion, device)
        train_acc.append(tr_acc)
        test_acc.append(te_acc)
        logging.info(f"Custom Layer {layer_type}, Epoch {epoch+1}: Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}")

    plot_curves(train_acc, test_acc, f"Custom Layer - {layer_type}", f"plots/custom_{layer_type}.png")
    logging.info(f"Layer {layer_type}, Total Params: {count_params(model)}")


blocks = {
    "basic": ResidualBlock,
    "bottleneck": BottleneckBlock,
    "wide": WideResidualBlock
}

for name, block in blocks.items():
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        block(16),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_acc, test_acc = [], []
    for epoch in range(EPOCHS):
        tr_acc, _ = train(model, train_loader, optimizer, criterion, device)
        te_acc, _ = evaluate(model, test_loader, criterion, device)
        train_acc.append(tr_acc)
        test_acc.append(te_acc)
        logging.info(f"Residual Block {name}, Epoch {epoch+1}: Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}")

    plot_curves(train_acc, test_acc, f"Residual Block - {name}", f"plots/resblock_{name}.png")
    logging.info(f"Block {name}, Total Params: {count_params(model)}")
