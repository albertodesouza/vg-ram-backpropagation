"""
MNIST data utilities with balanced sampling.

Supports three configurations:
  - tiny:  100 samples (10 per class) used for both train and test.
  - small: 900 train (90/class) + 100 test (10/class), balanced.
  - full:  50 000 train + 10 000 val + 10 000 test.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _balanced_indices(
    targets: torch.Tensor,
    samples_per_class: int,
    num_classes: int = 10,
    offset: int = 0,
) -> list[int]:
    """Return *samples_per_class* indices for each of *num_classes* classes.

    Args:
        targets:           1-D tensor of integer labels.
        samples_per_class: how many examples to pick per class.
        num_classes:       total number of classes.
        offset:            skip the first *offset* occurrences of each class
                           (useful for carving disjoint train / test splits
                           from the same dataset).

    Returns:
        Flat list of selected indices.
    """
    indices: list[int] = []
    for c in range(num_classes):
        class_idx = (targets == c).nonzero(as_tuple=True)[0].tolist()
        start = offset
        end = offset + samples_per_class
        if end > len(class_idx):
            raise ValueError(
                f"Class {c} has only {len(class_idx)} samples, "
                f"but offset={offset} + samples_per_class={samples_per_class} "
                f"requires {end}."
            )
        indices.extend(class_idx[start:end])
    return indices


def get_mnist_loaders(
    config: dict,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    """Build MNIST DataLoaders according to *config*.

    Args:
        config: the ``data`` section of the YAML configuration.
        data_dir: where to download / cache MNIST.

    Returns:
        (train_loader, val_loader_or_None, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0], shape (1, 28, 28)
    ])

    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    batch_size = config["batch_size"]
    train_samples = config.get("train_samples", len(train_set))
    val_samples = config.get("val_samples", 0)
    test_samples = config.get("test_samples", len(test_set))
    num_classes = 10
    same_train_test = config.get("same_train_test", False)

    total_from_train = train_samples + val_samples
    use_sequential = total_from_train >= len(train_set)

    if use_sequential:
        # Large-scale: split the training set sequentially (no per-class
        # balancing needed because the full MNIST training set is already
        # roughly balanced).
        all_idx = list(range(len(train_set)))
        train_idx = all_idx[:train_samples]
        train_subset = Subset(train_set, train_idx)

        val_loader: DataLoader | None = None
        if val_samples > 0:
            val_idx = all_idx[train_samples:train_samples + val_samples]
            val_subset = Subset(train_set, val_idx)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        # Small-scale: balanced sampling per class.
        per_class_train = train_samples // num_classes
        train_idx = _balanced_indices(
            train_set.targets, per_class_train, num_classes, offset=0,
        )
        train_subset = Subset(train_set, train_idx)

        val_loader = None
        if val_samples > 0:
            per_class_val = val_samples // num_classes
            val_idx = _balanced_indices(
                train_set.targets, per_class_val, num_classes,
                offset=per_class_train,
            )
            val_subset = Subset(train_set, val_idx)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # ----- Test subset -----
    if same_train_test:
        test_subset = train_subset
    elif test_samples < len(test_set):
        per_class_test = test_samples // num_classes
        test_idx = _balanced_indices(
            test_set.targets, per_class_test, num_classes, offset=0,
        )
        test_subset = Subset(test_set, test_idx)
    else:
        test_subset = Subset(test_set, list(range(len(test_set))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
