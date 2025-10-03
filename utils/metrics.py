import torch

def accuracy(preds, labels):
    _, predicted = preds.max(1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

