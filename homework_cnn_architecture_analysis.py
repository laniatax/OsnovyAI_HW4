import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
from models.cnn_models import CNNKernelVariants, CNNDepthVariants
from utils.training_utils import train, evaluate
from utils.visualization_utils import plot_curves, visualize_feature_maps, visualize_activations
from utils.comparison_utils import count_params

logging.basicConfig(level=logging.INFO)

transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_set = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5

# --- Kernel size experiment ---
kernels = ['3x3', '5x5', '7x7', '1x1+3x3']
for k in kernels:
    model = CNNKernelVariants(kernel_type=k, in_channels=1, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_acc, test_acc = [], []
    for epoch in range(EPOCHS):
        tr_acc, _ = train(model, train_loader, optimizer, criterion, device)
        te_acc, _ = evaluate(model, test_loader, criterion, device)
        train_acc.append(tr_acc)
        test_acc.append(te_acc)
        logging.info(f"Kernel {k}, Epoch {epoch+1}: Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}")

    plot_curves(train_acc, test_acc, f"MNIST Kernel {k}", f"plots/kernel_{k}_mnist.png")
    visualize_activations(model, test_loader, device, f"plots/activations_kernel_{k}.png")
    logging.info(f"Kernel {k}, Total Params: {count_params(model)}")

depths = ['shallow', 'medium', 'deep', 'residual']
for d in depths:
    model = CNNDepthVariants(depth=d, in_channels=1, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_acc, test_acc = [], []
    for epoch in range(EPOCHS):
        tr_acc, _ = train(model, train_loader, optimizer, criterion, device)
        te_acc, _ = evaluate(model, test_loader, criterion, device)
        train_acc.append(tr_acc)
        test_acc.append(te_acc)
        logging.info(f"Depth {d}, Epoch {epoch+1}: Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}")

    plot_curves(train_acc, test_acc, f"MNIST Depth {d}", f"plots/depth_{d}_mnist.png")
    visualize_feature_maps(model, test_loader, device, f"plots/featuremaps_{d}.png")
    logging.info(f"Depth {d}, Total Params: {count_params(model)}")
