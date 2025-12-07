import sys
from pathlib import Path

import torch

# add repo root so we can import from shared/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import mnist_dataloaders  # noqa: E402

def train(model, device, train_loader, optimizer, criterion, epoch):
    """train one epoch"""
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) 
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')


def test(model, device, test_loader, criterion): 
    """simple eval loop"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * len(data) 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item() 

    test_loss /= len(test_loader.dataset)

    print(f'\nTest: loss {test_loss:.4f}, acc {100. * correct / len(test_loader.dataset):.1f}% ({correct}/{len(test_loader.dataset)})')


def get_device():
    """pick gpu if we can"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    return device


def get_mnist_loaders(
    batch_size: int = 64,
    test_batch_size: int | None = None,
    normalize: bool = False,
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    """MNIST loaders via the shared helper."""
    return mnist_dataloaders(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        normalize=normalize,
        num_workers=num_workers,
        shuffle_train=shuffle_train,
    )
