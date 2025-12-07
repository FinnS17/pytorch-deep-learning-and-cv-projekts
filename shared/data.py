"""MNIST helpers shared by both projects."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Default to the repository-level data/ so both projects share the download
_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = _ROOT / "data"


def _mnist_transform(normalize: bool = False) -> transforms.Compose:
    """Basic MNIST transform."""
    ops = [transforms.ToTensor()]
    if normalize:
        # Standard MNIST mean/std
        ops.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(ops)


def load_mnist_numpy(
    data_dir: Path = DEFAULT_DATA_DIR,
    download: bool = True,
    flatten: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return MNIST as numpy arrays, optionally flattened/normalized."""
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=download, transform=transform)

    X_train = train_ds.data.numpy().astype(np.float32)
    X_test = test_ds.data.numpy().astype(np.float32)
    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    if flatten:
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)

    y_train = train_ds.targets.numpy().astype(np.int64)
    y_test = test_ds.targets.numpy().astype(np.int64)
    return X_train, y_train, X_test, y_test


def mnist_dataloaders(
    batch_size: int = 64,
    test_batch_size: int | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    download: bool = True,
    normalize: bool = False,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """PyTorch DataLoaders backed by the shared MNIST download."""
    transform = _mnist_transform(normalize=normalize)
    train_ds = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=download, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size or batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
