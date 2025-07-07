import matplotlib.pyplot as plt
import torch

def plot_curves(train_acc, test_acc, title, save_path):
    plt.figure()
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def visualize_feature_maps(model, loader, device, save_path):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        x = x.to(device)
        features = model.conv(x[:1])
        fig, axs = plt.subplots(1, min(features.shape[1], 8), figsize=(15, 3))
        for i in range(min(features.shape[1], 8)):
            axs[i].imshow(features[0, i].cpu(), cmap='gray')
            axs[i].axis('off')
        plt.savefig(save_path)
        plt.close()

def visualize_activations(model, loader, device, save_path):
    visualize_feature_maps(model, loader, device, save_path)
