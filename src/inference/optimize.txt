"""
Model Optimization for Deployment
===================================
Model quantization, pruning, and ONNX export
for efficient CPU inference.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.ao.quantization import get_default_qconfig, prepare, convert
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Optimize models for deployment on CPU.
    Supports quantization, pruning, and ONNX export.
    """

    def __init__(self, model: nn.Module, config: Dict):
        """
        Initialize optimizer.

        Args:
            model: PyTorch model to optimize
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = "cpu"  # Optimization for CPU deployment

    def quantize_dynamic(self) -> nn.Module:
        """
        Apply dynamic quantization (INT8).
        Best for models with LSTM/Linear layers.

        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization...")

        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )

        # Calculate size reduction
        original_size = self._get_model_size(self.model)
        quantized_size = self._get_model_size(quantized_model)

        logger.info(f"Original size: {original_size:.2f} MB")
        logger.info(f"Quantized size: {quantized_size:.2f} MB")
        logger.info(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")

        return quantized_model

    def quantize_static(
        self,
        calibration_data: torch.utils.data.DataLoader,
        num_calibration_batches: int = 100
    ) -> nn.Module:
        """
        Apply static quantization with calibration.
        Better accuracy than dynamic quantization.

        Args:
            calibration_data: DataLoader for calibration
            num_calibration_batches: Number of batches for calibration

        Returns:
            Quantized model
        """
        logger.info("Applying static quantization...")

        # Prepare model for quantization
        model = self.model.cpu()
        model.eval()

        # Set quantization config
        model.qconfig = get_default_qconfig('fbgemm')

        # Fuse modules where possible
        # Note: This requires model-specific fusion

        # Prepare for calibration
        prepared_model = prepare(model)

        # Calibration
        logger.info("Running calibration...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_data):
                if batch_idx >= num_calibration_batches:
                    break
                images = batch["image"]
                prepared_model(images)

        # Convert to quantized model
        quantized_model = convert(prepared_model)

        logger.info("Static quantization complete")
        return quantized_model

    def prune_model(
        self,
        amount: float = 0.3,
        method: str = "l1_unstructured"
    ) -> nn.Module:
        """
        Apply pruning to reduce model size.

        Args:
            amount: Fraction of connections to prune
            method: Pruning method

        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune

        logger.info(f"Applying {method} pruning with amount={amount}...")

        model = self.model.cpu()

        # Get all conv and linear layers
        layers_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers_to_prune.append((module, 'weight'))

        # Apply pruning
        if method == "l1_unstructured":
            prune.global_unstructured(
                layers_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        elif method == "random_unstructured":
            prune.global_unstructured(
                layers_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=amount
            )

        # Make pruning permanent
        for module, name in layers_to_prune:
            prune.remove(module, name)

        # Calculate sparsity
        total_params = 0
        zero_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        sparsity = zero_params / total_params * 100
        logger.info(f"Model sparsity: {sparsity:.1f}%")

        return model

    def export_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
        opset_version: int = 12
    ):
        """
        Export model to ONNX format.

        Args:
            output_path: Output file path
            input_shape: Input tensor shape (B, C, H, W)
            opset_version: ONNX opset version
        """
        logger.info("Exporting model to ONNX...")

        model = self.model.cpu()
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        logger.info(f"ONNX model saved: {output_path}")

        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verified successfully")
        except Exception as e:
            logger.warning(f"ONNX verification failed: {e}")

    def export_torchscript(
        self,
        output_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256)
    ):
        """
        Export model to TorchScript format.

        Args:
            output_path: Output file path
            input_shape: Input tensor shape
        """
        logger.info("Exporting model to TorchScript...")

        model = self.model.cpu()
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Trace model
        traced_model = torch.jit.trace(model, dummy_input)

        # Save
        traced_model.save(output_path)

        logger.info(f"TorchScript model saved: {output_path}")

    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb

    def benchmark_inference(
        self,
        model: nn.Module,
        input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model inference speed.

        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_iterations: Number of iterations
            warmup: Number of warmup iterations

        Returns:
            Dictionary with timing statistics
        """
        import time

        model = model.cpu()
        model.eval()

        dummy_input = torch.randn(*input_shape)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                times.append(end - start)

        times = np.array(times) * 1000  # Convert to ms

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": float(1000 / np.mean(times))
        }


class ONNXInference:
    """
    Run inference using ONNX Runtime for optimized CPU performance.
    """

    def __init__(self, model_path: str):
        """
        Initialize ONNX inference.

        Args:
            model_path: Path to ONNX model
        """
        import onnxruntime as ort

        # Set optimization options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        # Create session
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"ONNX session created: {model_path}")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on image.

        Args:
            image: Input image array (B, C, H, W)

        Returns:
            Prediction array
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: image}
        )

        return outputs[0]


def optimize_for_deployment(
    model_path: str,
    config_path: str,
    output_dir: str,
    quantize: bool = True,
    prune: bool = False,
    export_onnx: bool = True
):
    """
    Optimize trained model for deployment.

    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to configuration file
        output_dir: Output directory for optimized models
        quantize: Whether to apply quantization
        prune: Whether to apply pruning
        export_onnx: Whether to export to ONNX
    """
    import yaml
    from ..models import load_model

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load model
    model = load_model(model_path, config, "cpu")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize optimizer
    optimizer = ModelOptimizer(model, config)

    # Benchmark original model
    logger.info("\nBenchmarking original model...")
    original_stats = optimizer.benchmark_inference(model)
    logger.info(f"Original: {original_stats['mean_ms']:.2f}ms ({original_stats['fps']:.1f} FPS)")

    # Apply optimizations
    optimized_model = model

    if prune:
        optimized_model = optimizer.prune_model(amount=0.3)

    if quantize:
        optimized_model = optimizer.quantize_dynamic()

        # Benchmark quantized model
        logger.info("\nBenchmarking quantized model...")
        quant_stats = optimizer.benchmark_inference(optimized_model)
        logger.info(f"Quantized: {quant_stats['mean_ms']:.2f}ms ({quant_stats['fps']:.1f} FPS)")

        # Save quantized model
        quant_path = output_dir / "model_quantized.pth"
        torch.save(optimized_model.state_dict(), quant_path)
        logger.info(f"Quantized model saved: {quant_path}")

    if export_onnx:
        onnx_path = output_dir / "model.onnx"
        optimizer.export_onnx(str(onnx_path))

        # Export TorchScript
        ts_path = output_dir / "model_traced.pt"
        optimizer.export_torchscript(str(ts_path))

    logger.info("\nOptimization complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize model for deployment")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--prune", action="store_true", help="Apply pruning")
    parser.add_argument("--onnx", action="store_true", help="Export to ONNX")

    args = parser.parse_args()

    optimize_for_deployment(
        args.model,
        args.config,
        args.output,
        quantize=args.quantize,
        prune=args.prune,
        export_onnx=args.onnx
    )

