"""
dataset.py — Data loading and image augmentation pipelines.

The core idea of SimCLR is:
  For EACH image, generate TWO different random augmentations of it.
  These two augmented versions form a "positive pair".
  The model must learn to produce similar feature vectors for both,
  even though they look visually different.

This forces the backbone to learn *what the image is* (its content),
not *how it was transformed* (its exact appearance).
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# CIFAR-10 normalization constants
# These are the per-channel mean and standard deviation of the CIFAR-10
# training set. Normalizing inputs helps the optimizer converge faster.
# ---------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


class SimCLRTransform:
    """
    Wraps an augmentation pipeline to produce TWO random views of one image.

    When PyTorch's DataLoader calls `transform(image)`, it calls `__call__` below.
    We run the same random pipeline twice on the same source image, so we get
    two visually different crops/colors of the same content — the "positive pair".

    Why these specific augmentations?  The SimCLR paper found this combination
    most effective for making the model learn content-invariant features:
      - RandomResizedCrop  → learns position & scale invariance
      - RandomHorizontalFlip → learns left-right invariance
      - ColorJitter         → learns color/brightness invariance
      - RandomGrayscale     → prevents over-reliance on color alone
    """

    def __init__(self, image_size: int = 32):
        self.augment = transforms.Compose([
            # Randomly crop a portion of the image and resize back to original size.
            # scale=(0.2, 1.0) means the crop can be as small as 20% of the original area.
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),

            # Flip left-right with 50% probability.
            transforms.RandomHorizontalFlip(p=0.5),

            # Randomly change brightness, contrast, saturation, hue.
            # Applied with 80% probability; the four numbers control max distortion strength.
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.1)
            ], p=0.8),

            # Convert to grayscale (3 identical channels) with 20% probability.
            transforms.RandomGrayscale(p=0.2),

            # Convert PIL image to a [0,1] float tensor of shape [C, H, W].
            transforms.ToTensor(),

            # Normalize each channel: output = (input - mean) / std
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])

    def __call__(self, image):
        # Run the SAME pipeline twice; randomness makes the two outputs look different.
        view1 = self.augment(image)
        view2 = self.augment(image)
        return view1, view2   # Returns a tuple (not a single tensor)


# Standard evaluation transform: no augmentation, just convert and normalize.
# Used for kNN monitor and linear probing — we want clean, deterministic features.
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# Standard supervised augmentation: milder than SimCLR, preserves labels reliably.
SUPERVISED_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # pad 4px on each side then random crop
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


def get_ssl_loader(data_dir: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    """
    DataLoader for SimCLR SSL training.

    Each batch item is ((view1, view2), label).
    The label is ignored during SSL — we don't use it for the contrastive loss.
    drop_last=True ensures every batch has exactly `batch_size` items,
    which matters for the NT-Xent loss computation.
    """
    dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=SimCLRTransform(image_size=32)
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True
    )


def get_eval_loaders(data_dir: str, batch_size: int,
                     num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    """
    DataLoaders for evaluation (kNN monitor and linear probing).
    No augmentation — just clean, normalized images.
    Returns (train_loader, test_loader).
    """
    train_set = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=EVAL_TRANSFORM)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=EVAL_TRANSFORM)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_supervised_loaders(data_dir: str, batch_size: int,
                           num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    """
    DataLoaders for supervised learning training.
    Train split uses mild augmentation; test split uses no augmentation.
    Returns (train_loader, test_loader).
    """
    train_set = datasets.CIFAR10(root=data_dir, train=True,  download=True,
                                 transform=SUPERVISED_TRAIN_TRANSFORM)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True,
                                 transform=EVAL_TRANSFORM)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
