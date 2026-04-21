"""
main.py — Entry point for AI HW2: SimCLR Self-Supervised Learning on CIFAR-10.

This script runs two required experiments and saves results for your report:

  Experiment 1 (Required):  SimCLR SSL training
    - Trains the backbone + projector using contrastive loss (no labels used).
    - Tracks loss curve and kNN accuracy curve during training.
    - After training, runs linear probing to measure representation quality.

  Experiment 2 (Required):  Supervised learning baseline
    - Trains the same backbone + a linear head using cross-entropy loss (labels used).
    - Provides a direct accuracy comparison against SSL + linear probing.

Usage examples:
  python main.py                      # Run both experiments (default settings)
  python main.py --batch-size 128     # Smaller batch if GPU memory is limited
  python main.py --temperature 0.1    # Ablation: sharp temperature
  python main.py --temperature 5.0    # Ablation: flat temperature
  python main.py --skip-supervised    # Run SSL only (skips experiment 2)

Output files (in ./results/):
  ssl_curves.png    — Loss and kNN accuracy during SSL training
  sl_curves.png     — Loss and test accuracy during supervised training
  summary.txt       — Final accuracy numbers for your report table
"""

import argparse
import os
import torch
import matplotlib
matplotlib.use("Agg")   # Use non-interactive backend (works without a display)
import matplotlib.pyplot as plt

import config
from dataset  import get_ssl_loader, get_eval_loaders, get_supervised_loaders
from model    import SimCLRModel, SupervisedModel
from trainer  import train_simclr, train_supervised
from evaluate import linear_probing, evaluate_accuracy


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("  Apple Silicon GPU (MPS) detected.")
    else:
        dev = torch.device("cpu")
        print("  No GPU found.  Using CPU (training will be slow).")
    return dev


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_ssl_curves(history: dict, save_dir: str) -> None:
    """Save SSL training loss and kNN accuracy curves as a single PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("SimCLR SSL Training Curves", fontsize=14)

    # Left plot: NT-Xent loss over epochs
    axes[0].plot(range(1, len(history['loss']) + 1), history['loss'], color='steelblue')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NT-Xent Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # Right plot: kNN accuracy over epochs
    if history['knn_acc']:
        epochs, accs = zip(*history['knn_acc'])
        axes[1].plot(epochs, [a * 100 for a in accs],
                     color='darkorange', marker='o', markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("kNN Accuracy (%)")
        axes[1].set_title(f"kNN Monitor (k={config.KNN_K})")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "ssl_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_supervised_curves(history: dict, save_dir: str) -> None:
    """Save supervised learning loss and test accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Supervised Learning Training Curves", fontsize=14)

    axes[0].plot(range(1, len(history['loss']) + 1), history['loss'], color='steelblue')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    if history['test_acc']:
        epochs, accs = zip(*history['test_acc'])
        axes[1].plot(epochs, [a * 100 for a in accs], color='darkorange',
                     marker='o', markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Test Accuracy (%)")
        axes[1].set_title("Test Accuracy")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "sl_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI HW2: SimCLR on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch-size",      type=int,   default=None,
                        help="Override SSL_BATCH_SIZE from config.py")
    parser.add_argument("--temperature",     type=float, default=None,
                        help="Override TEMPERATURE from config.py (ablation study)")
    parser.add_argument("--ssl-epochs",      type=int,   default=None,
                        help="Override SSL_EPOCHS from config.py")
    parser.add_argument("--skip-supervised", action="store_true",
                        help="Skip the supervised learning baseline (Experiment 2)")
    parser.add_argument("--random-baseline", action="store_true",
                        help="Run Experiment 3: random frozen backbone (lower bound)")
    args = parser.parse_args()

    # Apply command-line overrides to config
    if args.batch_size  is not None: config.SSL_BATCH_SIZE = args.batch_size
    if args.temperature is not None: config.TEMPERATURE    = args.temperature
    if args.ssl_epochs  is not None: config.SSL_EPOCHS     = args.ssl_epochs

    # ------------------------------------------------------------------
    # Setup directories and device
    # ------------------------------------------------------------------
    os.makedirs(config.DATA_DIR,       exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR,    exist_ok=True)

    print("\n" + "="*65)
    print("  AI HW2: SimCLR Self-Supervised Learning on CIFAR-10")
    print(f"  Batch size : {config.SSL_BATCH_SIZE}")
    print(f"  Temperature: {config.TEMPERATURE}")
    print(f"  SSL epochs : {config.SSL_EPOCHS}")
    print("="*65)

    device = get_device()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nPreparing data loaders...")
    ssl_loader             = get_ssl_loader(config.DATA_DIR, config.SSL_BATCH_SIZE,
                                            config.NUM_WORKERS)
    eval_train_loader, \
    eval_test_loader       = get_eval_loaders(config.DATA_DIR, config.SSL_BATCH_SIZE,
                                              config.NUM_WORKERS)

    # ------------------------------------------------------------------
    # Experiment 1: SimCLR SSL Training + Linear Probing
    # ------------------------------------------------------------------
    print("\n" + "="*65)
    print("  EXPERIMENT 1: SimCLR Self-Supervised Learning")
    print("="*65)

    ssl_model = SimCLRModel(
        pretrained_backbone=False,    # Train entirely from scratch
        projector_hidden=config.PROJECTOR_HIDDEN_DIM,
        projector_out=config.PROJECTOR_OUT_DIM,
    ).to(device)

    ssl_history = train_simclr(
        model=ssl_model,
        ssl_loader=ssl_loader,
        eval_train_loader=eval_train_loader,
        eval_test_loader=eval_test_loader,
        device=device,
    )

    # Save the trained model so you can reload it later without retraining
    ssl_ckpt = os.path.join(config.CHECKPOINT_DIR, "simclr.pt")
    torch.save(ssl_model.state_dict(), ssl_ckpt)
    print(f"\n  Checkpoint saved: {ssl_ckpt}")

    # Plot training curves
    print("\n  Saving training curves...")
    plot_ssl_curves(ssl_history, config.RESULTS_DIR)

    # Linear probing: freeze backbone, train one linear layer
    print("\n" + "-"*65)
    print("  Linear Probing (evaluating SSL representation quality)")
    print("-"*65)
    ssl_lp_acc = linear_probing(
        model=ssl_model,
        train_loader=eval_train_loader,
        test_loader=eval_test_loader,
        device=device,
        num_classes=config.NUM_CLASSES,
        epochs=config.LP_EPOCHS,
        lr=config.LP_LR,
        weight_decay=config.LP_WEIGHT_DECAY,
    )

    # ------------------------------------------------------------------
    # Experiment 2: Supervised Learning Baseline
    # ------------------------------------------------------------------
    sl_final_acc = None
    if not args.skip_supervised:
        print("\n" + "="*65)
        print("  EXPERIMENT 2: Supervised Learning Baseline")
        print("="*65)

        sl_train_loader, sl_test_loader = get_supervised_loaders(
            config.DATA_DIR, config.SSL_BATCH_SIZE, config.NUM_WORKERS
        )

        sl_model = SupervisedModel(num_classes=config.NUM_CLASSES).to(device)
        sl_history = train_supervised(
            model=sl_model,
            train_loader=sl_train_loader,
            test_loader=sl_test_loader,
            device=device,
        )

        sl_ckpt = os.path.join(config.CHECKPOINT_DIR, "supervised.pt")
        torch.save(sl_model.state_dict(), sl_ckpt)
        print(f"\n  Checkpoint saved: {sl_ckpt}")

        plot_supervised_curves(sl_history, config.RESULTS_DIR)

        sl_final_acc = evaluate_accuracy(sl_model, sl_test_loader, device)

    # ------------------------------------------------------------------
    # Experiment 3: Random Frozen Backbone (lower bound)
    # ------------------------------------------------------------------
    random_lp_acc = None
    if args.random_baseline:
        print("\n" + "="*65)
        print("  EXPERIMENT 3: Random Frozen Backbone (lower bound)")
        print("  No training — linear probing on top of random features")
        print("="*65)

        # Create a model with random weights and skip training entirely.
        # This shows the floor: how well a linear layer can do when the
        # features it receives are completely meaningless.
        random_model = SimCLRModel(
            pretrained_backbone=False,
            projector_hidden=config.PROJECTOR_HIDDEN_DIM,
            projector_out=config.PROJECTOR_OUT_DIM,
        ).to(device)

        random_lp_acc = linear_probing(
            model=random_model,
            train_loader=eval_train_loader,
            test_loader=eval_test_loader,
            device=device,
            num_classes=config.NUM_CLASSES,
            epochs=config.LP_EPOCHS,
            lr=config.LP_LR,
            weight_decay=config.LP_WEIGHT_DECAY,
        )

    # ------------------------------------------------------------------
    # Print and save summary
    # ------------------------------------------------------------------
    print("\n" + "="*65)
    print("  RESULTS SUMMARY")
    print("="*65)
    if random_lp_acc is not None:
        print(f"  Random Frozen Backbone        : {random_lp_acc*100:.2f}%  ← lower bound")
    print(f"  SSL + Linear Probing          : {ssl_lp_acc*100:.2f}%")
    if sl_final_acc is not None:
        print(f"  Supervised Learning           : {sl_final_acc*100:.2f}%  ← upper bound")
    print("="*65)

    # Write summary to a text file for easy reference when writing your report
    summary_path = os.path.join(config.RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Temperature         : {config.TEMPERATURE}\n")
        f.write(f"Batch size          : {config.SSL_BATCH_SIZE}\n")
        f.write(f"SSL epochs          : {config.SSL_EPOCHS}\n")
        if random_lp_acc is not None:
            f.write(f"Random frozen acc   : {random_lp_acc*100:.2f}%\n")
        f.write(f"SSL+LinearProbe acc : {ssl_lp_acc*100:.2f}%\n")
        if sl_final_acc is not None:
            f.write(f"Supervised acc      : {sl_final_acc*100:.2f}%\n")
    print(f"\n  Summary written to: {summary_path}")
    print("\nDone.  Check ./results/ for plots and ./checkpoints/ for saved models.")


if __name__ == "__main__":
    main()
