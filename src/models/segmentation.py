"""
Lightweight Segmentation Models for Feature Extraction
=======================================================
Memory-efficient models optimized for low-resource systems.
Uses MobileNetV2/V3 encoders with U-Net decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import segmentation_models_pytorch as smp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net model using MobileNetV2 encoder.
    Optimized for systems with limited RAM and no GPU.
    """

    def __init__(
        self,
        encoder_name: str = "mobilenetv2_100",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 7,
        activation: Optional[str] = None
    ):
        """
        Initialize model.

        Args:
            encoder_name: Name of encoder backbone
            encoder_weights: Pretrained weights to use
            in_channels: Number of input channels
            num_classes: Number of output classes
            activation: Activation function for output
        """
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )

        # Log model info
        self._log_model_info()

    def _log_model_info(self):
        """Log model parameters count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Model: Lightweight U-Net with {type(self.model.encoder).__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (FP32)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_encoder(self):
        """Freeze encoder weights for fine-tuning."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder weights frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder weights unfrozen")


class LightweightDeepLabV3(nn.Module):
    """
    Lightweight DeepLabV3+ model using MobileNetV2 encoder.
    Better for large objects like buildings and roads.
    """

    def __init__(
        self,
        encoder_name: str = "mobilenetv2_100",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 7,
        activation: Optional[str] = None
    ):
        """
        Initialize model.

        Args:
            encoder_name: Name of encoder backbone
            encoder_weights: Pretrained weights to use
            in_channels: Number of input channels
            num_classes: Number of output classes
            activation: Activation function for output
        """
        super().__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )

        self._log_model_info()

    def _log_model_info(self):
        """Log model parameters count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Model: DeepLabV3+ with {self.model.encoder.name}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiTaskSegmentationModel(nn.Module):
    """
    Multi-task model for simultaneous:
    1. Semantic segmentation (building, road, water, background)
    2. Building roof classification (RCC, Tiled, Tin, Others)

    Uses shared encoder with task-specific decoders.
    """

    def __init__(
        self,
        encoder_name: str = "mobilenetv2_100",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_seg_classes: int = 4,  # background, building, road, water
        num_roof_classes: int = 4,  # RCC, Tiled, Tin, Others
    ):
        """
        Initialize multi-task model.

        Args:
            encoder_name: Name of encoder backbone
            encoder_weights: Pretrained weights to use
            in_channels: Number of input channels
            num_seg_classes: Number of segmentation classes
            num_roof_classes: Number of roof type classes
        """
        super().__init__()

        # Shared encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )

        encoder_channels = self.encoder.out_channels

        # Segmentation decoder
        self.seg_decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None
        )
        self.seg_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=num_seg_classes,
            activation=None,
            kernel_size=3
        )

        # Roof classification decoder (same structure but different output)
        self.roof_decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None
        )
        self.roof_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=num_roof_classes,
            activation=None,
            kernel_size=3
        )

        self._log_model_info()

    def _log_model_info(self):
        """Log model parameters count."""
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Multi-task Model initialized")
        logger.info(f"Total parameters: {total_params:,}")

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Dictionary with 'segmentation' and 'roof_type' predictions
        """
        # Shared encoding
        features = self.encoder(x)

        # Segmentation branch
        seg_decoder_out = self.seg_decoder(*features)
        seg_out = self.seg_head(seg_decoder_out)

        # Roof classification branch
        roof_decoder_out = self.roof_decoder(*features)
        roof_out = self.roof_head(roof_decoder_out)

        return {
            "segmentation": seg_out,
            "roof_type": roof_out
        }


class FeatureExtractor(nn.Module):
    """
    Combined feature extraction model that handles:
    - Building footprint with roof classification
    - Road extraction
    - Water body extraction

    Single unified model with 7 output classes:
    0: Background
    1: Building (RCC roof)
    2: Building (Tiled roof)
    3: Building (Tin roof)
    4: Building (Other roof)
    5: Road
    6: Water body
    """

    def __init__(
        self,
        config: Dict,
        pretrained: bool = True
    ):
        """
        Initialize feature extractor.

        Args:
            config: Model configuration dictionary
            pretrained: Whether to use pretrained weights
        """
        super().__init__()

        model_config = config.get("model", {}).get("segmentation", {})

        encoder_name = model_config.get("encoder_name", "mobilenetv2_100")
        encoder_weights = "imagenet" if pretrained else None
        in_channels = model_config.get("in_channels", 3)
        num_classes = model_config.get("classes", 7)
        decoder = model_config.get("decoder", "unet")

        if decoder == "unet":
            self.model = LightweightUNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
        elif decoder == "deeplabv3plus":
            self.model = LightweightDeepLabV3(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder}")

        self.num_classes = num_classes
        self.class_names = [
            "background",
            "building_rcc",
            "building_tiled",
            "building_tin",
            "building_others",
            "road",
            "waterbody"
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


def create_model(config: Dict) -> nn.Module:
    """
    Factory function to create model based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Model instance
    """
    return FeatureExtractor(config, pretrained=True)


# def load_model(
#     model_path: str,
#     config: Dict,
#     device: str = "cpu"
# ) -> nn.Module:
#     """
#     Load a trained model from checkpoint.
#
#     Args:
#         model_path: Path to checkpoint file
#         config: Model configuration
#         device: Device to load model on
#
#     Returns:
#         Loaded model
#     """
#     model = create_model(config)
#     model.model.freeze_encoder()
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#
#     if "model_state_dict" in checkpoint:
#         model.load_state_dict(checkpoint["model_state_dict"])
#     else:
#         model.load_state_dict(checkpoint)
#
#     model.to(device)
#     model.eval()
#
#     logger.info(f"Model loaded from {model_path}")
#     return model
def load_model(
    model_path: str,
    config: Dict,
    device: str = "cpu"
) -> nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to checkpoint file
        config: Model configuration
        device: Device to load model on

    Returns:
        Loaded model
    """
    model = create_model(config)
    model.model.freeze_encoder()

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint  # assume it's already a state dict

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    logger.info(f"Model loaded from {model_path}")
    return model

if __name__ == "__main__":
    # Test model creation
    config = {
        "model": {
            "segmentation": {
                "encoder_name": "mobilenetv2_100",
                "encoder_weights": "imagenet",
                "decoder": "unet",
                "in_channels": 3,
                "classes": 7
            }
        }
    }

    model = create_model(config)

    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test prediction
    pred = model.predict(x)
    print(f"Prediction shape: {pred.shape}")

