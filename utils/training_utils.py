import torch

def train(model, loader, optimizer, criterion, device):
    model.train()
    correct = total = loss_sum = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)

    return correct / total, loss_sum / total

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = loss_sum = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)

    return correct / total, loss_sum / total
