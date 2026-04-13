import os
from typing import Dict, Tuple, Any
import torch.nn as nn

from src.models.segmentation import load_model as load_pytorch_model
from src.inference.onnx_inference import load_onnx_model


def load_inference_model(
    model_path: str,
    config: Dict,
    device: str = "cpu"
) -> Tuple[Any, str]:
    """
    Load either a PyTorch or ONNX model based on file extension.

    Returns:
        Tuple of (model, model_type)
    """
    ext = os.path.splitext(model_path)[1].lower()

    if ext == ".onnx":
        model = load_onnx_model(model_path)
        return model, "onnx"

    elif ext in [".pth", ".pt"]:
        model = load_pytorch_model(model_path, config, device)
        return model, "pytorch"

    else:
        raise ValueError(f"Unsupported model format: {ext}")