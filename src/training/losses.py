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
import logging

logger = logging.getLogger(__name__)


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
    Enhanced version with per-class alpha weighting.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        per_class_alpha: bool = False
    ):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.alpha = alpha
        self.per_class_alpha = per_class_alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Logits (B, C, H, W)
            target: Ground truth (B, H, W)
        """
        # Get cross-entropy loss
        ce_loss = F.cross_entropy(
            pred,
            target,
            weight=self.class_weights.to(pred.device)
            if self.class_weights is not None else None,
            reduction='none'
        )

        # Compute focal weight: (1 - pt)^gamma
        # pt is the probability of the true class
        p = torch.exp(-ce_loss)
        focal_weight = (1.0 - p) ** self.gamma
        
        # Apply focal weighting
        focal_loss = self.alpha * focal_weight * ce_loss

        return focal_loss.mean()




# ============================================================
# Lovasz Softmax Loss
# ============================================================

class LovaszSoftmax(nn.Module):
    """
    Lovasz-Softmax loss for semantic segmentation.
    Better for boundary preservation and multi-class scenarios.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -1
    ):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Logits (B, C, H, W)
            target: Ground truth (B, H, W)
        """
        return lovasz_softmax(pred, target, self.class_weights, self.ignore_index)


def lovasz_softmax(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = -1
) -> torch.Tensor:
    """Compute Lovasz softmax loss."""
    B, C, H, W = pred.shape
    
    # Flatten spatial dimensions
    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
    target_flat = target.reshape(-1)
    
    # Remove ignore_index
    if ignore_index >= 0:
        mask = target_flat != ignore_index
        pred_flat = pred_flat[mask]
        target_flat = target_flat[mask]
    
    # Per-class lovasz loss
    total_loss = 0.0
    for c in range(C):
        target_binary = (target_flat == c).float()
        pred_c = pred_flat[:, c]
        
        # Skip if no positive examples
        if target_binary.sum() == 0:
            continue
        
        # Sort predictions
        sorted_pred, sort_idx = torch.sort(pred_c, descending=True)
        sorted_target = target_binary[sort_idx]
        
        # Jaccard index
        intersection = torch.cumsum(sorted_target, dim=0)
        union = torch.arange(1, len(sorted_target) + 1, device=pred.device).float() + torch.sum(sorted_target) - intersection
        jaccard = 1.0 - intersection.float() / union
        
        # Lovasz loss
        loss_c = torch.mean(jaccard)
        
        # Apply class weight
        if class_weights is not None and c < len(class_weights):
            loss_c = loss_c * class_weights[c]
        
        total_loss = total_loss + loss_c
    
    return total_loss / max(C, 1.0)


# ============================================================
# Combined Segmentation Loss
# ============================================================

class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation with:
    - Focal Loss or Weighted Cross-Entropy
    - Weighted Dice Loss
    - Optional Label Smoothing
    - Optional LovaszSoftmax
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        lovasz_weight: float = 0.0,  # Set to 0.3 for additional regularization
        class_weights: Optional[List[float]] = None,
        use_focal: bool = True,
        use_label_smoothing: bool = True,
        label_smoothing: float = 0.1
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            # Ensure weights don't become too extreme (helps stability)
            class_weights = torch.clamp(class_weights, min=0.1, max=10.0)

        if use_focal:
            self.ce_loss = FocalLoss(
                alpha=0.25,
                gamma=2.0,
                class_weights=class_weights
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing if use_label_smoothing else 0.0,
                ignore_index=0
            )

        self.dice_loss = DiceLoss(class_weights=class_weights)
        
        # Optional Lovasz loss for better boundary handling
        if lovasz_weight > 0:
            self.lovasz_loss = LovaszSoftmax(class_weights=class_weights)
        else:
            self.lovasz_loss = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: Model predictions (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """

        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total = self.ce_weight * ce + self.dice_weight * dice
        
        results = {
            "total": total,
            "ce": ce,
            "dice": dice
        }
        
        # Add Lovasz loss if enabled
        if self.lovasz_loss is not None:
            lovasz = self.lovasz_loss(pred, target)
            total = total + self.lovasz_weight * lovasz
            results["lovasz"] = lovasz
            results["total"] = total

        return results


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
    method: str = "effective"
) -> torch.Tensor:
    """
    Calculate class weights to handle imbalance.
    
    Methods:
    - "inverse": Simple inverse frequency weighting
    - "sqrt_inverse": Square root of inverse frequency (less aggressive)
    - "effective": Effective number of samples (best for imbalanced datasets)
    - "binary": Binary cross-entropy style weighting
    """
    total = sum(class_counts.values())
    weights = torch.ones(num_classes)

    for cls in range(num_classes):
        count = class_counts.get(cls, 0)
        
        if count == 0:
            # Unseen class - use average weight
            weights[cls] = 1.0
        elif method == "inverse":
            # Simple inverse frequency
            weights[cls] = total / (count * num_classes)
        elif method == "sqrt_inverse":
            # Less aggressive than inverse
            weights[cls] = np.sqrt(total / (count * num_classes))
        elif method == "effective":
            # Effective number of samples (recommended)
            # Handles long-tail distribution better
            beta = 0.999
            effective_num = 1.0 - np.power(beta, count)
            if effective_num > 0:
                weights[cls] = (1.0 - beta) / effective_num
            else:
                weights[cls] = 1.0
        elif method == "binary":
            # Binary cross-entropy style
            weights[cls] = total / (2 * count)

    # Normalize to sum to num_classes (for stability)
    weights = weights / weights.sum() * num_classes
    
    # Clamp to prevent extreme values
    weights = torch.clamp(weights, min=0.1, max=10.0)
    
    logger.info(f"Class weights ({method}): {weights.tolist()}")
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