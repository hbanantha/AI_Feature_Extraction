"""
Object Detection Model for Infrastructure Features
===================================================
Lightweight object detection for:
- Distribution Transformers
- Overhead Tanks
- Wells

Uses YOLOv5 Nano or custom lightweight detector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))


class LightweightDetectorBackbone(nn.Module):
    """
    Lightweight backbone for object detection.
    Based on MobileNet-style architecture.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Initial convolution
        self.conv1 = ConvBlock(in_channels, 32, stride=2)

        # Depthwise separable blocks
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 64, stride=2)
        )

        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            DepthwiseSeparableConv(128, 128, stride=2)
        )

        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            DepthwiseSeparableConv(256, 256),
            DepthwiseSeparableConv(256, 256, stride=2)
        )

        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(256, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512, stride=2)
        )

    def forward(self, x):
        """Return multi-scale features."""
        x = self.conv1(x)      # /2
        c1 = self.block1(x)    # /4
        c2 = self.block2(c1)   # /8
        c3 = self.block3(c2)   # /16
        c4 = self.block4(c3)   # /32

        return [c2, c3, c4]


class DetectionHead(nn.Module):
    """Detection head for predicting boxes and classes."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Output channels: (x, y, w, h, objectness, class_scores) * num_anchors
        out_channels = num_anchors * (5 + num_classes)

        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class LightweightObjectDetector(nn.Module):
    """
    Lightweight object detector for infrastructure features.

    Detects:
    - Distribution Transformers
    - Overhead Tanks
    - Wells
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        anchors: Optional[List[List[Tuple[int, int]]]] = None
    ):
        """
        Initialize detector.

        Args:
            num_classes: Number of object classes
            in_channels: Number of input channels
            anchors: Anchor boxes for different scales
        """
        super().__init__()

        self.num_classes = num_classes
        self.class_names = [
            "distribution_transformer",
            "overhead_tank",
            "well"
        ]

        # Default anchors (will be optimized based on data)
        if anchors is None:
            self.anchors = [
                [(10, 13), (16, 30), (33, 23)],    # Small objects
                [(30, 61), (62, 45), (59, 119)],   # Medium objects
                [(116, 90), (156, 198), (373, 326)]  # Large objects
            ]
        else:
            self.anchors = anchors

        self.num_anchors = len(self.anchors[0])

        # Backbone
        self.backbone = LightweightDetectorBackbone(in_channels)

        # Feature Pyramid Network (FPN) style neck
        self.lateral_conv3 = nn.Conv2d(512, 256, 1)
        self.lateral_conv2 = nn.Conv2d(256, 256, 1)
        self.lateral_conv1 = nn.Conv2d(128, 256, 1)

        self.fpn_conv3 = DepthwiseSeparableConv(256, 256)
        self.fpn_conv2 = DepthwiseSeparableConv(256, 256)
        self.fpn_conv1 = DepthwiseSeparableConv(256, 256)

        # Detection heads for each scale
        self.head1 = DetectionHead(256, num_classes, self.num_anchors)
        self.head2 = DetectionHead(256, num_classes, self.num_anchors)
        self.head3 = DetectionHead(256, num_classes, self.num_anchors)

        self._log_model_info()

    def _log_model_info(self):
        """Log model information."""
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Lightweight Object Detector initialized")
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Total parameters: {total_params:,}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of detection tensors at different scales
        """
        # Get backbone features
        c2, c3, c4 = self.backbone(x)

        # FPN
        p4 = self.lateral_conv3(c4)
        p4 = self.fpn_conv3(p4)

        p3 = self.lateral_conv2(c3) + F.interpolate(
            p4, size=c3.shape[2:], mode='nearest'
        )
        p3 = self.fpn_conv2(p3)

        p2 = self.lateral_conv1(c2) + F.interpolate(
            p3, size=c2.shape[2:], mode='nearest'
        )
        p2 = self.fpn_conv1(p2)

        # Detection heads
        det1 = self.head1(p2)  # Small objects
        det2 = self.head2(p3)  # Medium objects
        det3 = self.head3(p4)  # Large objects

        return [det1, det2, det3]

    def decode_predictions(
        self,
        outputs: List[torch.Tensor],
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ) -> List[Dict]:
        """
        Decode raw outputs to bounding boxes.

        Args:
            outputs: List of raw detection outputs
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold

        Returns:
            List of detection dictionaries per image
        """
        batch_size = outputs[0].shape[0]
        all_detections = []

        for batch_idx in range(batch_size):
            boxes = []
            scores = []
            class_ids = []

            for scale_idx, output in enumerate(outputs):
                # Reshape output
                B, C, H, W = output.shape
                output = output[batch_idx].view(
                    self.num_anchors, 5 + self.num_classes, H, W
                ).permute(0, 2, 3, 1)

                # Extract predictions
                for anchor_idx in range(self.num_anchors):
                    anchor_pred = output[anchor_idx]

                    # Objectness
                    obj = torch.sigmoid(anchor_pred[..., 4])

                    # Class probabilities
                    class_probs = torch.sigmoid(anchor_pred[..., 5:])

                    # Combined confidence
                    conf = obj.unsqueeze(-1) * class_probs

                    # Find predictions above threshold
                    mask = conf > conf_threshold

                    if mask.any():
                        # Get box coordinates
                        for i in range(H):
                            for j in range(W):
                                for c in range(self.num_classes):
                                    if conf[i, j, c] > conf_threshold:
                                        # Decode box
                                        tx = torch.sigmoid(anchor_pred[i, j, 0])
                                        ty = torch.sigmoid(anchor_pred[i, j, 1])
                                        tw = anchor_pred[i, j, 2]
                                        th = anchor_pred[i, j, 3]

                                        # Get anchor
                                        anchor_w, anchor_h = self.anchors[scale_idx][anchor_idx]

                                        # Calculate actual coordinates
                                        cx = (j + tx.item()) / W
                                        cy = (i + ty.item()) / H
                                        w = (anchor_w * np.exp(tw.item())) / 256
                                        h = (anchor_h * np.exp(th.item())) / 256

                                        boxes.append([
                                            cx - w/2, cy - h/2,
                                            cx + w/2, cy + h/2
                                        ])
                                        scores.append(conf[i, j, c].item())
                                        class_ids.append(c)

            # Apply NMS
            if len(boxes) > 0:
                boxes = torch.tensor(boxes)
                scores = torch.tensor(scores)
                class_ids = torch.tensor(class_ids)

                # Simple NMS per class
                keep_indices = []
                for c in range(self.num_classes):
                    c_mask = class_ids == c
                    if c_mask.any():
                        c_boxes = boxes[c_mask]
                        c_scores = scores[c_mask]

                        keep = self._nms(c_boxes, c_scores, nms_threshold)
                        c_indices = torch.where(c_mask)[0][keep]
                        keep_indices.extend(c_indices.tolist())

                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                class_ids = class_ids[keep_indices]

                all_detections.append({
                    "boxes": boxes,
                    "scores": scores,
                    "class_ids": class_ids,
                    "class_names": [self.class_names[i] for i in class_ids]
                })
            else:
                all_detections.append({
                    "boxes": torch.tensor([]),
                    "scores": torch.tensor([]),
                    "class_ids": torch.tensor([]),
                    "class_names": []
                })

        return all_detections

    def _nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """Simple NMS implementation."""
        if len(boxes) == 0:
            return torch.tensor([])

        # Sort by score
        _, indices = scores.sort(descending=True)

        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current.item())

            if len(indices) == 1:
                break

            # Calculate IoU with rest
            current_box = boxes[current]
            rest_boxes = boxes[indices[1:]]

            ious = self._box_iou(current_box.unsqueeze(0), rest_boxes)

            # Keep boxes with IoU below threshold
            mask = ious.squeeze() < threshold
            indices = indices[1:][mask]

        return torch.tensor(keep)

    def _box_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between boxes."""
        # Intersection
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)

        # Union
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        union = area1 + area2 - inter_area

        return inter_area / (union + 1e-6)


def create_detector(config: Dict) -> LightweightObjectDetector:
    """
    Create object detector from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Detector instance
    """
    det_config = config.get("model", {}).get("detection", {})

    return LightweightObjectDetector(
        num_classes=det_config.get("classes", 3),
        in_channels=config.get("data", {}).get("input_channels", 3)
    )


if __name__ == "__main__":
    # Test detector
    detector = LightweightObjectDetector(num_classes=3)

    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    outputs = detector(x)

    print("Detection outputs:")
    for i, out in enumerate(outputs):
        print(f"  Scale {i}: {out.shape}")

