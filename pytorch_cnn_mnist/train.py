from pathlib import Path

import torch
from torch import nn, optim

from cnn_model import Cnn
from train_utils import get_device, get_mnist_loaders, test, train


def main():
    ROOT = Path(__file__).resolve().parent
    device = get_device()
    model = Cnn().to(device)

    train_loader, test_loader = get_mnist_loaders(batch_size=64, normalize=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    epochs = 6

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

    ckpt_path = ROOT / "cnn_mnist.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved weights to {ckpt_path}")


if __name__ == "__main__":
    main()
