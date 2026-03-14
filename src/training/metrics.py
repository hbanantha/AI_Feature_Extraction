"""
Evaluation Metrics for Feature Extraction
Segmentation and detection metrics for model evaluation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


# =============================================================================
# Segmentation Metrics
# =============================================================================
class SegmentationMetrics:
    """
    Metrics for semantic segmentation evaluation.
    Computes IoU, Dice, Precision, Recall, F1, Accuracy.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -1,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [
            f"class_{i}" for i in range(num_classes)
        ]
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64,
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update confusion matrix with a batch."""

        if pred.dim() == 4:
            pred = pred.argmax(dim=1)

        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        for p, t in zip(pred, target):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                self.confusion_matrix[t, p] += 1

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {}

        iou_per_class = []
        dice_per_class = []
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp

            iou = tp / (tp + fp + fn + 1e-10)
            dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            iou_per_class.append(iou)
            dice_per_class.append(dice)
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

            metrics[f"iou_{self.class_names[i]}"] = iou
            metrics[f"dice_{self.class_names[i]}"] = dice
            metrics[f"precision_{self.class_names[i]}"] = precision
            metrics[f"recall_{self.class_names[i]}"] = recall
            metrics[f"f1_{self.class_names[i]}"] = f1

        metrics["mIoU"] = np.mean(iou_per_class)
        metrics["mDice"] = np.mean(dice_per_class)
        metrics["mPrecision"] = np.mean(precision_per_class)
        metrics["mRecall"] = np.mean(recall_per_class)
        metrics["mF1"] = np.mean(f1_per_class)

        total_correct = np.trace(self.confusion_matrix)
        total_samples = self.confusion_matrix.sum()
        metrics["accuracy"] = total_correct / (total_samples + 1e-10)

        # Optional: building-specific aggregation
        building_classes = [1, 2, 3, 4]
        building_tp = sum(self.confusion_matrix[i, i] for i in building_classes)

        building_fp = sum(
            self.confusion_matrix[:, i].sum() - self.confusion_matrix[i, i]
            for i in building_classes
        )

        building_fn = sum(
            self.confusion_matrix[i, :].sum() - self.confusion_matrix[i, i]
            for i in building_classes
        )

        metrics["building_iou"] = building_tp / (
            building_tp + building_fp + building_fn + 1e-10
        )
        metrics["building_precision"] = building_tp / (
            building_tp + building_fp + 1e-10
        )
        metrics["building_recall"] = building_tp / (
            building_tp + building_fn + 1e-10
        )

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        return self.confusion_matrix.copy()


# =============================================================================
# Detection Metrics
# =============================================================================
class DetectionMetrics:
    """
    Metrics for object detection evaluation.
    Computes AP, mAP, Precision, Recall.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        iou_threshold: float = 0.5,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [
            f"class_{i}" for i in range(num_classes)
        ]
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.predictions = defaultdict(list)
        self.total_gt = defaultdict(int)

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_classes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
    ):
        for cls in gt_classes.cpu().numpy():
            self.total_gt[int(cls)] += 1

        if len(pred_boxes) == 0:
            return

        gt_matched = set()
        sorted_indices = torch.argsort(pred_scores, descending=True)

        for idx in sorted_indices:
            pred_box = pred_boxes[idx]
            pred_cls = int(pred_classes[idx].item())
            pred_score = pred_scores[idx].item()

            is_tp = False

            for gt_idx, (gt_box, gt_cls) in enumerate(
                zip(gt_boxes, gt_classes)
            ):
                if gt_idx in gt_matched:
                    continue
                if int(gt_cls.item()) != pred_cls:
                    continue

                iou = self._compute_iou(pred_box, gt_box)
                if iou >= self.iou_threshold:
                    is_tp = True
                    gt_matched.add(gt_idx)
                    break

            self.predictions[pred_cls].append((pred_score, is_tp))

    def _compute_iou(
        self,
        box1: torch.Tensor,
        box2: torch.Tensor,
    ) -> float:

        x1 = max(box1[0].item(), box2[0].item())
        y1 = max(box1[1].item(), box2[1].item())
        x2 = min(box1[2].item(), box2[2].item())
        y2 = min(box1[3].item(), box2[3].item())

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = area1.item() + area2.item() - inter_area

        return inter_area / (union_area + 1e-10)

    def compute(self) -> Dict[str, float]:
        metrics = {}
        ap_per_class = []

        for cls in range(self.num_classes):
            predictions = sorted(
                self.predictions[cls], key=lambda x: -x[0]
            )
            total_gt = self.total_gt[cls]

            if total_gt == 0:
                ap_per_class.append(0.0)
                continue

            tp_cumsum = 0
            fp_cumsum = 0
            precision_list = []
            recall_list = []

            for score, is_tp in predictions:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1

                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / total_gt

                precision_list.append(precision)
                recall_list.append(recall)

            ap = self._compute_ap(precision_list, recall_list)
            ap_per_class.append(ap)

            metrics[f"AP_{self.class_names[cls]}"] = ap

            if precision_list:
                metrics[f"precision_{self.class_names[cls]}"] = precision_list[-1]
                metrics[f"recall_{self.class_names[cls]}"] = recall_list[-1]
            else:
                metrics[f"precision_{self.class_names[cls]}"] = 0.0
                metrics[f"recall_{self.class_names[cls]}"] = 0.0

        metrics["MAP"] = np.mean(ap_per_class) if ap_per_class else 0.0

        return metrics

    def _compute_ap(
        self,
        precision_list: List[float],
        recall_list: List[float],
    ) -> float:

        if not precision_list:
            return 0.0

        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            precisions = [
                p for p, r in zip(precision_list, recall_list) if r >= t
            ]
            if precisions:
                ap += max(precisions) / 11.0

        return ap


# =============================================================================
# Combined Metrics
# =============================================================================
class CombinedMetrics:
    """
    Combined metrics for segmentation and detection.
    """

    def __init__(
        self,
        seg_num_classes: int = 7,
        det_num_classes: int = 3,
        seg_class_names: Optional[List[str]] = None,
        det_class_names: Optional[List[str]] = None,
    ):
        self.seg_metrics = SegmentationMetrics(
            seg_num_classes, seg_class_names
        )
        self.det_metrics = DetectionMetrics(
            det_num_classes, det_class_names
        )

    def reset(self):
        self.seg_metrics.reset()
        self.det_metrics.reset()

    def update_segmentation(self, pred, target):
        self.seg_metrics.update(pred, target)

    def update_detection(
        self,
        pred_boxes,
        pred_scores,
        pred_classes,
        gt_boxes,
        gt_classes,
    ):
        self.det_metrics.update(
            pred_boxes,
            pred_scores,
            pred_classes,
            gt_boxes,
            gt_classes,
        )

    def compute(self) -> Dict[str, float]:
        seg_metrics = self.seg_metrics.compute()
        det_metrics = self.det_metrics.compute()

        combined = {}

        for k, v in seg_metrics.items():
            combined[f"seg/{k}"] = v

        for k, v in det_metrics.items():
            combined[f"det/{k}"] = v

        combined["overall_score"] = (
            0.7 * seg_metrics.get("mIoU", 0.0)
            + 0.3 * det_metrics.get("MAP", 0.0)
        )

        return combined