"""
Visualization Utilities
Tools for visualizing predictions, creating comparisons, and generating reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Color mapping for classes
# ---------------------------------------------------------

CLASS_COLORS = {
    0: (0, 0, 0),        # Background
    1: (255, 0, 0),      # Building RCC
    2: (0, 255, 0),      # Building Tiled
    3: (0, 0, 255),      # Building Tin
    4: (255, 255, 0),    # Building Others
    5: (128, 128, 128),  # Road
    6: (0, 255, 255),    # Water body
}

CLASS_NAMES = [
    "Background",
    "Building (RCC)",
    "Building (Tiled)",
    "Building (Tin)",
    "Building (Others)",
    "Road",
    "Waterbody",
]

# ---------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------

def colorize_prediction(
    prediction: np.ndarray,
    colors: Dict[int, Tuple[int, int, int]] = CLASS_COLORS,
) -> np.ndarray:
    """
    Convert class prediction to RGB image.

    Args:
        prediction: Class prediction array (H, W)
        colors: Color mapping dictionary

    Returns:
        RGB image (H, W, 3)
    """
    h, w = prediction.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in colors.items():
        mask = prediction == class_idx
        rgb[mask] = color

    return rgb


def create_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create overlay of prediction on original image.

    Args:
        image: Original RGB image (H, W, 3)
        prediction: Class prediction (H, W)
        alpha: Overlay transparency

    Returns:
        Overlay image (H, W, 3)
    """
    colored_pred = colorize_prediction(prediction)

    # Do not overlay background
    mask = prediction > 0

    overlay = image.copy()
    overlay[mask] = cv2.addWeighted(
        colored_pred[mask],
        alpha,
        image[mask],
        1 - alpha,
        0,
    )

    return overlay


def plot_comparison(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Comparison",
):
    """
    Create side-by-side comparison plot.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground truth
    gt_colored = colorize_prediction(ground_truth)
    axes[1].imshow(gt_colored)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Prediction
    pred_colored = colorize_prediction(prediction)
    axes[2].imshow(pred_colored)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # Overlay
    overlay = create_overlay(image, prediction)
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=np.array(color) / 255.0, label=CLASS_NAMES[idx])
        for idx, color in CLASS_COLORS.items()
    ]

    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=len(CLASS_NAMES),
        fontsize=10,
    )

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history: List[Dict],
    save_path: Optional[str] = None,
):
    """
    Plot training history curves.
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["val_loss"] for h in history]
    miou = [h["val"]["mIoU"] for h in history]
    accuracy = [h["val"]["accuracy"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(epochs, train_loss, label="Train Loss", marker="o")
    axes[0].plot(epochs, val_loss, label="Val Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Metrics plot
    axes[1].plot(epochs, miou, label="mIoU", marker="o")
    axes[1].plot(epochs, accuracy, label="Accuracy", marker="o")
    axes[1].axhline(y=0.95, color="r", linestyle="--", label="Target (95%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = None,
):
    """
    Plot normalized confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    cm_norm = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    im = ax.imshow(cm_norm, cmap="Blues")
    plt.colorbar(im)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm_norm[i, j]
            color = "white" if value > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved: {save_path}")
    else:
        plt.show()

    plt.close()


def create_report_figures(
    output_dir: str,
    history_path: Optional[str] = None,
    confusion_matrix: Optional[np.ndarray] = None,
    sample_predictions: Optional[List[Dict]] = None,
):
    """
    Create all report figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    if history_path and Path(history_path).exists():
        with open(history_path) as f:
            history = json.load(f)

        plot_training_history(
            history,
            save_path=str(output_dir / "training_history.png"),
        )

    # Confusion matrix
    if confusion_matrix is not None:
        plot_confusion_matrix(
            confusion_matrix,
            save_path=str(output_dir / "confusion_matrix.png"),
        )

    # Sample predictions
    if sample_predictions:
        for i, sample in enumerate(sample_predictions[:5]):
            plot_comparison(
                sample["image"],
                sample["ground_truth"],
                sample["prediction"],
                save_path=str(output_dir / f"sample_{i+1}.png"),
                title=f"Sample {i+1}",
            )

    logger.info(f"Report figures saved to: {output_dir}")