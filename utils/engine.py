import torch
from utils.losses import get_loss
from utils.metrics import accuracy

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    loss_fn = get_loss()
    total_loss, total_acc = 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)

    print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f} | Acc: {total_acc/len(loader):.4f}")

def evaluate(model, loader, device):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            total_acc += accuracy(outputs, labels)
    print(f"Validation Acc: {total_acc/len(loader):.4f}")

