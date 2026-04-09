"""
Model Evaluation Module for Semantic Segmentation
=================================================
Computes IoU, mIoU, Accuracy, Precision, Recall, and F1-score.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from src.models.segmentation import create_model
from src.preprocessing.dataloader import (
    DroneImageDataset,
    get_validation_augmentation,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ==============================================================
# Metrics
# ==============================================================

def compute_iou(conf_matrix: np.ndarray) -> np.ndarray:
    """Compute Intersection over Union for each class."""
    intersection = np.diag(conf_matrix)
    union = (
        conf_matrix.sum(axis=1)
        + conf_matrix.sum(axis=0)
        - intersection
    )
    return intersection / (union + 1e-10)


def compute_metrics(conf_matrix: np.ndarray) -> Dict:
    """Compute evaluation metrics from confusion matrix."""
    iou = compute_iou(conf_matrix)
    miou = np.nanmean(iou)
    accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()

    return {
        "IoU_per_class": iou.tolist(),
        "mIoU": float(miou),
        "Accuracy": float(accuracy),
    }


# ==============================================================
# Evaluator Class
# ==============================================================

class SegmentationEvaluator:
    """Evaluator for semantic segmentation models."""

    def __init__(self, config: Dict, model_path: str):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = create_model(config)
        checkpoint = torch.load(model_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded from {model_path}")

        # Number of classes
        self.num_classes = len(
            config["data"]["segmentation_classes"]
        )

    def create_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        transform = get_validation_augmentation(self.config)

        dataset = DroneImageDataset(
            tiles_dir=self.config["data"]["tiles_dir"],
            masks_dir=self.config["data"]["annotations_dir"],
            transform=transform,
            is_training=False,
            split_ratio=self.config["data"].get("split_ratio", 0.8),
            split_seed=self.config["data"].get("split_seed", 42),
        )

        logger.info(f"Evaluation dataset size: {len(dataset)}")

        return DataLoader(
            dataset,
            batch_size=self.config["training"].get("batch_size", 4),
            shuffle=False,
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=False,
        )

    def evaluate(self) -> Dict:
        """Run evaluation."""
        dataloader = self.create_dataloader()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy().flatten())
                all_targets.append(masks.cpu().numpy().flatten())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        # Confusion Matrix
        conf_matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(self.num_classes))
        )

        metrics = compute_metrics(conf_matrix)

        # Additional metrics
        metrics["Precision"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["Recall"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["F1-Score"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["Confusion_Matrix"] = conf_matrix.tolist()

        return metrics

    def save_results(self, metrics: Dict, output_path: str):
        """Save evaluation results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Evaluation results saved to {output_path}")


# ==============================================================
# Entry Function
# ==============================================================

def run_evaluation(config_path: str, model_path: str, output_path: str):
    """Run evaluation from configuration."""
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    evaluator = SegmentationEvaluator(config, model_path)
    metrics = evaluator.evaluate()
    evaluator.save_results(metrics, output_path)

    logger.info("Evaluation Complete!")
    logger.info(json.dumps(metrics, indent=4))