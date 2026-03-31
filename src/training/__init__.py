"""
Training Package Initialization
"""

from .losses import (
    DiceLoss,
    FocalLoss,
    LovaszSoftmax,
    CombinedSegmentationLoss,
    DetectionLoss,
    EWCLoss,
    get_class_weights,
)

from .metrics import (
    SegmentationMetrics,
    DetectionMetrics,
    CombinedMetrics,
)

from .trainer import (
    IncrementalTrainer,
    train,
)

from .tpu_utils import (
    TPUTrainingContext,
    TPUGradientAccumulator,
    optimize_model_for_tpu,
    create_tpu_compatible_dataloader,
    reduce_metrics_across_tpu,
)

__all__ = [
    "DiceLoss",
    "FocalLoss",
    "LovaszSoftmax",
    "CombinedSegmentationLoss",
    "DetectionLoss",
    "EWCLoss",
    "get_class_weights",
    "SegmentationMetrics",
    "DetectionMetrics",
    "CombinedMetrics",
    "IncrementalTrainer",
    "train",
    "TPUTrainingContext",
    "TPUGradientAccumulator",
    "optimize_model_for_tpu",
    "create_tpu_compatible_dataloader",
    "reduce_metrics_across_tpu",
]