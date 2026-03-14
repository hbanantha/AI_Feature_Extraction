"""
Loss Functions for Feature Extraction

Combined loss functions for segmentation and detection tasks.
Includes class weighting for imbalanced datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import numpy as np


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    Handles class imbalance better than cross-entropy.
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        num_classes = pred.shape[1]

        pred_soft = F.softmax(pred, dim=1)

        target_one_hot = F.one_hot(
            target, num_classes
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = torch.sum(pred_soft * target_one_hot, dim=dims)
        cardinality = torch.sum(pred_soft + target_one_hot, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )

        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
            dice_score = dice_score * weights

        return 1.0 - dice_score.mean()


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    Focuses on hard examples.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        ce_loss = F.cross_entropy(
            pred,
            target,
            weight=self.class_weights.to(pred.device)
            if self.class_weights is not None else None,
            reduction='none'
        )

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# ============================================================
# Combined Segmentation Loss
# ============================================================

class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation:
    - Cross-Entropy / Focal
    - Dice
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[List[float]] = None,
        use_focal: bool = True
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        if use_focal:
            self.ce_loss = FocalLoss(class_weights=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        self.dice_loss = DiceLoss(class_weights=class_weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)

        total = self.ce_weight * ce + self.dice_weight * dice

        return {
            "total": total,
            "ce": ce,
            "dice": dice
        }


# ============================================================
# Detection Loss (Simplified Placeholder)
# ============================================================

class DetectionLoss(nn.Module):
    """
    Simplified detection loss placeholder.
    """

    def __init__(
        self,
        num_classes: int = 3,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:

        total_loss = 0.0

        for pred in predictions:
            total_loss += pred.abs().mean() * 0.001

        total_loss = torch.tensor(
            total_loss,
            requires_grad=True
        )

        return {
            "total": total_loss,
            "coord": torch.tensor(0.0),
            "obj": torch.tensor(0.0),
            "cls": torch.tensor(0.0)
        }


# ============================================================
# Elastic Weight Consolidation (EWC)
# ============================================================

class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation (EWC)
    Prevents catastrophic forgetting.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 0.4
    ):
        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.params = {}
        self.fisher = {}

    def compute_fisher(
        self,
        model: nn.Module,
        data_loader,
        device: str = "cpu",
        num_samples: int = 200
    ):

        model.eval()

        fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        sample_count = 0

        for batch in data_loader:
            if sample_count >= num_samples:
                break

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            model.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, masks)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

            sample_count += images.size(0)

        for n in fisher:
            fisher[n] /= max(sample_count, 1)

        self.fisher = fisher
        self.params = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def forward(self, model: nn.Module) -> torch.Tensor:

        if not self.fisher:
            return torch.tensor(0.0)

        loss = 0.0

        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (
                    self.fisher[n] *
                    (p - self.params[n]).pow(2)
                ).sum()

        return self.lambda_ewc * loss


# ============================================================
# Class Weight Utility
# ============================================================

def get_class_weights(
    class_counts: Dict[int, int],
    num_classes: int,
    method: str = "inverse"
) -> torch.Tensor:

    total = sum(class_counts.values())
    weights = torch.zeros(num_classes)

    for cls, count in class_counts.items():
        if count > 0:
            if method == "inverse":
                weights[cls] = total / count
            elif method == "sqrt_inverse":
                weights[cls] = np.sqrt(total / count)
            elif method == "effective":
                beta = 0.999
                effective_num = 1.0 - np.power(beta, count)
                weights[cls] = (1.0 - beta) / effective_num

    weights = weights / weights.sum() * num_classes
    return weights


# ============================================================
# Main Test Block
# ============================================================

if __name__ == "__main__":

    # Dummy segmentation test
    B, C, H, W = 2, 3, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))

    loss_fn = CombinedSegmentationLoss()
    losses = loss_fn(pred, target)

    print("Segmentation Loss Test:")
    print(losses)

    # Dummy EWC test
    model = nn.Conv2d(3, 3, 3, padding=1)
    ewc = EWCLoss(model)
    penalty = ewc(model)

    print("\nEWC Penalty:", penalty)