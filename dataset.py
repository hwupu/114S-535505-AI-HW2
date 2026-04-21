"""
dataset.py — Data loading and augmentation for multiple datasets.

Supported datasets:
  cifar10    — 32×32, 10 classes,  50k train /  10k test
  stl10      — 96×96, 10 classes,   5k train /   8k test  (+ 100k unlabeled for SSL)
  flowers102 — variable,102 classes, 1k train /  6k test
  food101    — variable,101 classes,75k train /  25k test

All images are resized to 32×32 to match the modified ResNet-18 backbone.
Non-CIFAR datasets use ImageNet normalization statistics.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std':  (0.2023, 0.1994, 0.2010),
    },
    'stl10': {
        'num_classes': 10,
        'mean': (0.485, 0.456, 0.406),   # ImageNet stats
        'std':  (0.229, 0.224, 0.225),
    },
    'flowers102': {
        'num_classes': 102,
        'mean': (0.485, 0.456, 0.406),
        'std':  (0.229, 0.224, 0.225),
    },
    'food101': {
        'num_classes': 101,
        'mean': (0.485, 0.456, 0.406),
        'std':  (0.229, 0.224, 0.225),
    },
}

TARGET_SIZE = 32   # All datasets are resized to this resolution


def get_dataset_config(name: str) -> dict:
    """Return the config dict for a dataset, with a helpful error if unknown."""
    name = name.lower()
    if name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Supported: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[name]


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

class SimCLRTransform:
    """
    Applies two independent random augmentations to the same image.
    Returns (view1, view2) — a positive pair for contrastive learning.

    Includes an initial Resize so images of any input resolution are
    brought to TARGET_SIZE before augmentation crops are applied.
    """
    def __init__(self, mean, std, image_size: int = TARGET_SIZE):
        self.augment = transforms.Compose([
            # Resize first so RandomResizedCrop works consistently
            # regardless of the original image resolution.
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.augment(image), self.augment(image)


def _eval_transform(mean, std, image_size: int = TARGET_SIZE):
    """Clean transform for evaluation: resize, center-crop, normalize."""
    return transforms.Compose([
        transforms.Resize(image_size + 4),        # slightly larger to allow clean center crop
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _supervised_train_transform(mean, std, image_size: int = TARGET_SIZE):
    """Standard augmentation for supervised training."""
    return transforms.Compose([
        transforms.Resize(image_size + 4),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Dataset loader helper
# ---------------------------------------------------------------------------

def _load_split(dataset_name: str, data_dir: str, split: str, transform):
    """
    Load a named split of a dataset.

    split values per dataset:
      cifar10    : 'train' or 'test'
      stl10      : 'train', 'test', 'unlabeled', or 'train+unlabeled'
      flowers102 : 'train', 'val', or 'test'
      food101    : 'train' or 'test'
    """
    name = dataset_name.lower()
    if name == 'cifar10':
        return datasets.CIFAR10(
            root=data_dir, train=(split == 'train'),
            download=True, transform=transform
        )
    elif name == 'stl10':
        return datasets.STL10(
            root=data_dir, split=split,
            download=True, transform=transform
        )
    elif name == 'flowers102':
        return datasets.Flowers102(
            root=data_dir, split=split,
            download=True, transform=transform
        )
    elif name == 'food101':
        return datasets.Food101(
            root=data_dir, split=split,
            download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'")


# ---------------------------------------------------------------------------
# Public loader functions
# ---------------------------------------------------------------------------

def get_ssl_loader(dataset_name: str, data_dir: str,
                   batch_size: int, num_workers: int = 4) -> DataLoader:
    """
    DataLoader for SimCLR SSL training.
    Each item is ((view1, view2), label) — labels are ignored during SSL.

    STL-10 special case: uses 'train+unlabeled' (105k images) for a richer
    pool of negative samples, which benefits contrastive learning.
    """
    cfg = get_dataset_config(dataset_name)
    transform = SimCLRTransform(mean=cfg['mean'], std=cfg['std'])

    split = 'train+unlabeled' if dataset_name.lower() == 'stl10' else 'train'
    dataset = _load_split(dataset_name, data_dir, split, transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True, pin_memory=True)


def get_eval_loaders(dataset_name: str, data_dir: str,
                     batch_size: int, num_workers: int = 4) -> tuple:
    """
    Clean (no augmentation) DataLoaders for kNN monitor and linear probing.
    Returns (train_loader, test_loader).
    """
    cfg = get_dataset_config(dataset_name)
    transform = _eval_transform(cfg['mean'], cfg['std'])

    train_set = _load_split(dataset_name, data_dir, 'train', transform)
    test_set  = _load_split(dataset_name, data_dir, 'test',  transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_supervised_loaders(dataset_name: str, data_dir: str,
                           batch_size: int, num_workers: int = 4) -> tuple:
    """
    DataLoaders for supervised learning baseline.
    Train split uses mild augmentation; test split uses no augmentation.
    Returns (train_loader, test_loader).
    """
    cfg = get_dataset_config(dataset_name)
    train_transform = _supervised_train_transform(cfg['mean'], cfg['std'])
    test_transform  = _eval_transform(cfg['mean'], cfg['std'])

    train_set = _load_split(dataset_name, data_dir, 'train', train_transform)
    test_set  = _load_split(dataset_name, data_dir, 'test',  test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
