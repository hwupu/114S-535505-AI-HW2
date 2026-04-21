"""
config.py — All hyperparameters in one place.

Keeping all settings here makes it easy to run ablation experiments:
just change a value here instead of hunting through multiple files.
"""

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Choose which dataset to use for all experiments.
# Supported: 'cifar10', 'stl10', 'flowers102', 'food101'
#
# Notes:
#   - All datasets are resized to 32x32 to match the modified ResNet-18.
#   - STL-10 uses 105k images (5k labeled + 100k unlabeled) for SSL training.
#   - Flowers-102 / Food-101 are for transfer learning: train SSL on CIFAR-10
#     first, then run linear probing on these datasets.
DATASET = 'cifar10'

# Number of classes — derived automatically from DATASET.
# If you add a custom dataset, set this manually.
_CLASS_COUNTS = {'cifar10': 10, 'stl10': 10, 'flowers102': 102, 'food101': 101}
NUM_CLASSES = _CLASS_COUNTS.get(DATASET, 10)

DATA_DIR = "./data"

# ---------------------------------------------------------------------------
# SSL Training (SimCLR)
# ---------------------------------------------------------------------------
SSL_EPOCHS = 200
SSL_BATCH_SIZE = 256         # Try 128 or 64 if you run out of GPU memory
SSL_LR = 3e-4                # Learning rate for Adam optimizer
SSL_WEIGHT_DECAY = 1e-6      # L2 regularization strength

# ---------------------------------------------------------------------------
# SimCLR-specific settings
# ---------------------------------------------------------------------------
# Temperature controls how "sharp" the similarity distribution is:
#   - Low  (e.g. 0.1): model is very confident, but may overfit to easy pairs
#   - High (e.g. 5.0): model is very uncertain, loss signal is weak
#   - 0.5 is a good starting point (from the original SimCLR paper)
TEMPERATURE = 0.5

# Projector head dimensions: backbone output → hidden → projector output
PROJECTOR_HIDDEN_DIM = 512
PROJECTOR_OUT_DIM = 128

# ---------------------------------------------------------------------------
# kNN Monitor (runs periodically during SSL training to track progress)
# ---------------------------------------------------------------------------
KNN_K = 20                   # Number of nearest neighbors to use
KNN_INTERVAL = 5             # Run kNN evaluation every N epochs
KNN_TEMPERATURE = 0.1        # Softmax temperature for kNN weighted voting

# ---------------------------------------------------------------------------
# Linear Probing (evaluates SSL representation quality after training)
# ---------------------------------------------------------------------------
LP_EPOCHS = 100
LP_LR = 1e-3
LP_WEIGHT_DECAY = 1e-6
# Batch size reuses SSL_BATCH_SIZE, as specified in the assignment

# ---------------------------------------------------------------------------
# Supervised Learning Baseline
# ---------------------------------------------------------------------------
SL_EPOCHS = 200
SL_LR = 3e-4
SL_WEIGHT_DECAY = 1e-6

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
NUM_WORKERS = 4              # DataLoader worker threads; set to 0 on Windows if errors occur
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
