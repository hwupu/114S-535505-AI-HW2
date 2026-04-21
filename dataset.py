"""
dataset.py — Data loading and augmentation for multiple datasets.

Supported datasets:
  cifar10    — 32×32, 10 classes,  50k train /  10k test
  stl10      — 96×96, 10 classes,   5k train /   8k test  (+ 100k unlabeled for SSL)
  flowers102 — variable,102 classes, 1k train /  6k test
  food101    — variable,101 classes,75k train /  25k test

All images are resized to 32×32 to match the modified ResNet-18 backbone.
Non-CIFAR datasets use ImageNet normalization statistics.

Supported SSL augmentation strategies (pass name to get_ssl_loader):
  crop    — RandomResizedCrop + flip (spatial only)
  cutout  — flip + RandomErasing (occlusion)
  color   — ColorJitter + grayscale (colour only)
  sobel   — Sobel edge filter (replaces pixel values with edge magnitudes)
  noise   — Gaussian additive noise
  blur    — GaussianBlur
  rotate  — RandomRotation ± 30° + flip
  full    — crop + rotation + flip + colour + grayscale + blur + noise + cutout
  best    — crop + flip + colour + grayscale  ← SimCLR paper recommendation (default)
"""

import torch
import torch.nn.functional as F
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
# Custom tensor-level transforms used by some augmentation strategies
# ---------------------------------------------------------------------------

class _GaussianNoise:
    """Add isotropic Gaussian noise to a tensor (applied after ToTensor)."""
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std


class _SobelTransform:
    """
    Replace pixel content with Sobel edge magnitudes.

    Pipeline: convert to grayscale → apply Sobel kernels → L2 magnitude →
    normalize to [0, 1] → repeat across 3 channels.

    This is an intentionally weak augmentation: two random crops of the same
    image will share the same edge structure, so the model gets very little
    variation to learn from. Contrast with 'best' (colour + crop) to see
    why the original SimCLR paper found colour distortion critical.

    Note: dataset-specific normalization is not applied here because Sobel
    replaces the image content entirely; the output is already in [0, 1].
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        gray = tensor.mean(dim=0, keepdim=True)          # [1, H, W]
        gray4d = gray.unsqueeze(0)                        # [1, 1, H, W]

        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3)

        ex = F.conv2d(gray4d, kx, padding=1)
        ey = F.conv2d(gray4d, ky, padding=1)
        edges = (ex ** 2 + ey ** 2).sqrt().squeeze(0)    # [1, H, W]
        edges = edges / (edges.max() + 1e-8)              # normalize to [0, 1]
        return edges.repeat(3, 1, 1)                      # [3, H, W]


# ---------------------------------------------------------------------------
# Augmentation factory
# ---------------------------------------------------------------------------

AUGMENTATION_NAMES = [
    'crop', 'cutout', 'color', 'sobel', 'noise', 'blur', 'rotate', 'full', 'best'
]


def _build_augmentation(name: str, mean, std, image_size: int) -> transforms.Compose:
    """
    Return a Compose pipeline for the requested augmentation strategy.
    Both views of a SimCLR pair use the same pipeline (applied independently).
    """
    name = name.lower()
    normalize = transforms.Normalize(mean=mean, std=std)

    if name == 'crop':
        # Spatial variation only — no colour change.
        # Tests whether positional invariance alone is enough.
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

    elif name == 'cutout':
        # Random rectangular occlusion — tests robustness to missing patches.
        # RandomErasing is torchvision's built-in Cutout (operates on tensors).
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=1.0, scale=(0.1, 0.33),
                                     ratio=(0.3, 3.3), value=0),
        ])

    elif name == 'color':
        # Colour variation only — no crop.
        # Tests whether colour invariance alone is enough.
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    elif name == 'sobel':
        # Edge-map only — dataset normalization is skipped because Sobel replaces
        # pixel content; output is already normalised to [0, 1].
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            _SobelTransform(),
        ])

    elif name == 'noise':
        # Additive Gaussian noise only.
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
            _GaussianNoise(std=0.05),
        ])

    elif name == 'blur':
        # Gaussian smoothing only — removes fine texture, keeps global structure.
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize,
        ])

    elif name == 'rotate':
        # Random rotation ± 30° — tests rotational invariance.
        # Not in the original SimCLR paper; expected to underperform 'best'.
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

    elif name == 'full':
        # Kitchen-sink: crop + rotation + flip + colour + grayscale + blur +
        # noise + cutout. More invariances, but signal may become too noisy.
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
            _GaussianNoise(std=0.02),
        ])

    elif name == 'best':
        # SimCLR paper recommendation: crop + flip + colour + grayscale.
        # This is the default and should achieve the highest accuracy.
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    else:
        raise ValueError(
            f"Unknown augmentation '{name}'. "
            f"Supported: {AUGMENTATION_NAMES}"
        )


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

class SimCLRTransform:
    """
    Applies two independent random augmentations to the same image.
    Returns (view1, view2) — a positive pair for contrastive learning.

    augmentation: one of AUGMENTATION_NAMES (default 'best').
    """
    def __init__(self, mean, std, image_size: int = TARGET_SIZE,
                 augmentation: str = 'best'):
        self.augment = _build_augmentation(augmentation, mean, std, image_size)

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
                   batch_size: int, num_workers: int = 4,
                   augmentation: str = 'best') -> DataLoader:
    """
    DataLoader for SimCLR SSL training.
    Each item is ((view1, view2), label) — labels are ignored during SSL.

    augmentation: one of AUGMENTATION_NAMES (default 'best').
    STL-10 special case: uses 'train+unlabeled' (105k images) for a richer
    pool of negative samples, which benefits contrastive learning.
    """
    cfg = get_dataset_config(dataset_name)
    transform = SimCLRTransform(mean=cfg['mean'], std=cfg['std'],
                                augmentation=augmentation)

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
