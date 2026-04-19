"""
config.py — All hyperparameters in one place.

Keeping all settings here makes it easy to run ablation experiments:
just change a value here instead of hunting through multiple files.
"""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = "./data"          # CIFAR-10 will be auto-downloaded here
NUM_CLASSES = 10             # CIFAR-10 has 10 classes

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
