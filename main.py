"""
Main Entry Point for AI Feature Extraction Pipeline

Command-line interface for all operations:
- Data preprocessing (tiling)
- Model training
- Feature extraction (inference)
- Model optimization
"""

import argparse
import sys
import yaml
import logging
from pathlib import Path


# -------------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------------
# Preprocess Command
# -------------------------------------------------------------------------
def cmd_preprocess(args):
    """Run preprocessing (tiling) pipeline."""
    from src.preprocessing import GeoTIFFTiler, BatchTileProcessor

    config = load_config(args.config)

    tiler = GeoTIFFTiler(
        tile_size=config["data"]["tile_size"],
        overlap=config["data"]["tile_overlap"],
        min_valid_ratio=config["data"]["min_valid_pixels"],
        output_format="npy",
        memory_limit_gb=config["hardware"]["memory_limit_gb"],
    )

    if args.input_dir:
        # Process all files in directory
        processor = BatchTileProcessor(
            tiler,
            villages_per_batch=config["training"]["incremental"]["villages_per_batch"],
        )
        processor.process_all_villages(args.input_dir, args.output_dir)

    else:
        # Process single file
        tiler.process_geotiff(args.input_file, args.output_dir)

    logger.info("Preprocessing complete!")


# -------------------------------------------------------------------------
# Train Command
# -------------------------------------------------------------------------
def cmd_train(args):
    """Run training pipeline."""
    from src.training import train

    # train(args.config)
    train(args.config, resume_checkpoint=args.resume)

    logger.info("Training complete!")


# -------------------------------------------------------------------------
# Evaluation Command
# -------------------------------------------------------------------------
def cmd_evaluate(args):
    from src.evaluation.evaluator import run_evaluation

    run_evaluation(
        args.config,
        args.model,
        args.output,
    )
# -------------------------------------------------------------------------
# Inference Command
# -------------------------------------------------------------------------
def cmd_inference(args):
    """Run inference pipeline."""
    from src.inference import run_inference

    run_inference(
        args.config,
        args.model,
        args.input,
        args.output,
    )

    logger.info("Inference complete!")


# -------------------------------------------------------------------------
# Optimize Command
# -------------------------------------------------------------------------
def cmd_optimize(args):
    """Run model optimization."""
    from src.inference import optimize_for_deployment

    optimize_for_deployment(
        args.model,
        args.config,
        args.output,
        quantize=args.quantize,
        prune=args.prune,
        export_onnx=args.onnx,
    )

    logger.info("Optimization complete!")


# -------------------------------------------------------------------------
# Evaluate Command
# -------------------------------------------------------------------------
# def cmd_evaluate(args):
#     """Run model evaluation."""
#     from src.training import SegmentationMetrics
#     from src.models import load_model
#     from src.preprocessing import (
#         DroneImageDataset,
#         get_validation_augmentation,
#     )
#     from torch.utils.data import DataLoader
#     import torch
#
#     config = load_config(args.config)
#
#     # Load model
#     model = load_model(args.model, config, device="cpu")
#     model.eval()
#
#     # Create validation dataset
#     transform = get_validation_augmentation(config)
#
#     dataset = DroneImageDataset(
#         tiles_dir=args.data_dir,
#         masks_dir=args.masks_dir,
#         transform=transform,
#         is_training=False,
#     )
#
#     dataloader = DataLoader(
#         dataset,
#         batch_size=config["training"]["batch_size"],
#         shuffle=False,
#         num_workers=config["training"]["num_workers"],
#     )
#
#     # Metrics
#     metrics = SegmentationMetrics(
#         num_classes=config["data"]["num_seg_classes"],
#         class_names=list(config["data"]["segmentation_classes"].values()),
#     )
#
#     # Evaluation Loop
#     with torch.no_grad():
#         for batch in dataloader:
#             images = batch["image"]
#             masks = batch["mask"]
#
#             outputs = model(images)
#             metrics.update(outputs, masks)
#
#     metrics.print_report()


# -------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Feature Extraction from Drone Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

# Preprocess (tile) a single TIFF file
python main.py preprocess --config configs/config.yaml \
    --input-file data/raw/village1.tif \
    --output-dir data/tiles

# Preprocess all TIFF files in a directory
python main.py preprocess --config configs/config.yaml \
    --input-dir data/raw \
    --output-dir data/tiles

# Train the model
python main.py train --config configs/config.yaml

# Run Evaluation
python main.py evaluate --config configs/config.yaml \
    --model outputs/checkpoints/best_model.pth

# Run inference
python main.py inference --config configs/config.yaml \
    --model outputs/checkpoints/best_model.pth \
    --input data/test/village.tif \
    --output outputs/predictions

# Optimize model
python main.py optimize --config configs/config.yaml \
    --model outputs/checkpoints/best_model.pth \
    --output outputs/optimized \
    --quantize --onnx
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Tile large GeoTIFF files"
    )
    preprocess_parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    preprocess_parser.add_argument("--input-file", type=str, help="Single input TIFF")
    preprocess_parser.add_argument("--input-dir", type=str, help="Directory of TIFFs")
    preprocess_parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    preprocess_parser.set_defaults(func=cmd_preprocess)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    train_parser.set_defaults(func=cmd_train)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    parser_eval = subparsers.add_parser(
        "evaluate", help="Evaluate trained model"
    )
    parser_eval.add_argument(
        "--config", required=True, help="Path to config file"
    )
    parser_eval.add_argument(
        "--model", required=True, help="Path to trained model"
    )
    parser_eval.add_argument(
        "--output",
        default="outputs/evaluation/evaluation_metrics.json",
        help="Output metrics file",
    )
    parser_eval.set_defaults(func=cmd_evaluate)
    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    inference_parser = subparsers.add_parser(
        "inference", help="Run feature extraction"
    )
    inference_parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    inference_parser.add_argument(
        "--model", type=str, required=True, help="Model checkpoint path"
    )
    inference_parser.add_argument(
        "--input", type=str, required=True, help="Input TIFF file or directory"
    )
    inference_parser.add_argument(
        "--output", type=str, help="Output directory"
    )
    inference_parser.set_defaults(func=cmd_inference)

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------
    optimize_parser = subparsers.add_parser(
        "optimize", help="Optimize model for deployment"
    )
    optimize_parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    optimize_parser.add_argument(
        "--model", type=str, required=True, help="Model checkpoint path"
    )
    optimize_parser.add_argument(
        "--output", type=str, required=True, help="Output directory"
    )
    optimize_parser.add_argument(
        "--quantize", action="store_true", help="Apply INT8 quantization"
    )
    optimize_parser.add_argument(
        "--prune", action="store_true", help="Apply weight pruning"
    )
    optimize_parser.add_argument(
        "--onnx", action="store_true", help="Export to ONNX format"
    )
    optimize_parser.set_defaults(func=cmd_optimize)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    # evaluate_parser = subparsers.add_parser(
    #     "evaluate", help="Evaluate model performance"
    # )
    # evaluate_parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="configs/config.yaml",
    #     help="Path to config file",
    # )
    # evaluate_parser.add_argument(
    #     "--model", type=str, required=True, help="Model checkpoint path"
    # )
    # evaluate_parser.add_argument(
    #     "--data-dir", type=str, required=True, help="Directory with test tiles"
    # )
    # evaluate_parser.add_argument(
    #     "--masks-dir", type=str, required=True, help="Directory with ground truth masks"
    # )
    # evaluate_parser.set_defaults(func=cmd_evaluate)

    # ------------------------------------------------------------------
    # Parse Arguments
    # ------------------------------------------------------------------
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


# -------------------------------------------------------------------------
# Entry
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()