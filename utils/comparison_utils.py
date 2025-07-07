import torch
import time

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, loader, device, iterations=10):
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)

    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(x)
    end = time.time()
    avg_time = (end - start) / iterations
    print(f"Inference time per batch: {avg_time:.4f} sec")
