import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
from models.fc_models import FullyConnectedNet
from models.cnn_models import SimpleCNN, CNNWithResidual
from utils.training_utils import train, evaluate
from utils.visualization_utils import plot_curves
from utils.comparison_utils import count_params, measure_inference_time

logging.basicConfig(level=logging.INFO)

transform = transforms.Compose([transforms.ToTensor()])
train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_mnist, batch_size=64, shuffle=True)
test_loader = DataLoader(test_mnist, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5

models = {
    "FC": FullyConnectedNet([784, 256, 128, 10]),
    "SimpleCNN": SimpleCNN(in_channels=1, num_classes=10),
    "CNNResidual": CNNWithResidual(in_channels=1, num_classes=10)
}

for name, model in models.items():
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_accs, test_accs = [], []

    for epoch in range(EPOCHS):
        train_acc, _ = train(model, train_loader, optimizer, criterion, device)
        test_acc, _ = evaluate(model, test_loader, criterion, device)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        logging.info(f"{name} - Epoch {epoch+1}: Train={train_acc:.4f}, Test={test_acc:.4f}")

    plot_curves(train_accs, test_accs, title=name, save_path=f"plots/mnist_{name}.png")
    logging.info(f"{name} Params: {count_params(model)}")
    measure_inference_time(model, test_loader, device)
