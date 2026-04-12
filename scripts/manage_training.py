"""
Training Management Utility
============================
Helper script for managing training checkpoints, resuming interrupted training,
and monitoring training progress.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_checkpoints(checkpoint_dir: str) -> None:
    """List all available training checkpoints with metadata."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    
    if not checkpoints:
        logger.info("No checkpoints found")
        return
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    logger.info(f"Found {len(checkpoints)} checkpoints:")
    logger.info("-" * 80)
    
    for ckpt in checkpoints:
        mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
        size_mb = ckpt.stat().st_size / (1024**2)
        logger.info(f"  {ckpt.name:40s} | Size: {size_mb:6.2f}MB | Modified: {mtime}")
    
    logger.info("-" * 80)


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """Get the latest checkpoint for resuming training."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Prefer *_best.pth files, then get latest
    best_checkpoints = list(checkpoint_dir.glob("*_best.pth"))
    if best_checkpoints:
        latest = max(best_checkpoints, key=lambda x: x.stat().st_mtime)
        logger.info(f"Latest best checkpoint: {latest.name}")
        return str(latest)
    
    # Fall back to any checkpoint
    all_checkpoints = list(checkpoint_dir.glob("*.pth"))
    if all_checkpoints:
        latest = max(all_checkpoints, key=lambda x: x.stat().st_mtime)
        logger.info(f"Latest checkpoint: {latest.name}")
        return str(latest)
    
    logger.warning("No checkpoints found")
    return None


def show_training_history(log_dir: str, num_recent: int = 20) -> None:
    """Display recent training history."""
    log_dir = Path(log_dir)
    history_path = log_dir / "training_history.json"
    
    if not history_path.exists():
        logger.warning(f"Training history not found: {history_path}")
        return
    
    with open(history_path) as f:
        history = json.load(f)
    
    logger.info(f"Training History (showing last {num_recent} entries):")
    logger.info("-" * 100)
    logger.info(f"{'Epoch':>6} {'Batch':>6} {'Train Loss':>12} {'Val Loss':>12} {'mIoU':>10}")
    logger.info("-" * 100)
    
    for entry in history[-num_recent:]:
        epoch = entry.get("epoch", "N/A")
        batch = entry.get("batch_idx", "N/A")
        train_loss = entry.get("train_loss", 0.0)
        val_loss = entry.get("val_loss", 0.0)
        miou = entry.get("mIoU", 0.0)
        
        logger.info(f"{epoch:>6d} {batch:>6d} {train_loss:>12.4f} {val_loss:>12.4f} {miou:>10.4f}")
    
    logger.info("-" * 100)
    
    # Show best metrics
    if history:
        best_miou_entry = max(history, key=lambda x: x.get("mIoU", 0.0))
        logger.info(f"\nBest mIoU: {best_miou_entry['mIoU']:.4f} (Epoch {best_miou_entry['epoch']})")


def show_inference_status(output_dir: str) -> None:
    """Show status of incomplete inference jobs."""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        logger.warning(f"Output directory not found: {output_dir}")
        return
    
    checkpoints = list(output_dir.glob("*_inference_checkpoint.json"))
    
    if not checkpoints:
        logger.info("No incomplete inference jobs found")
        return
    
    logger.info(f"Found {len(checkpoints)} incomplete inference jobs:")
    logger.info("-" * 80)
    
    for ckpt in sorted(checkpoints):
        with open(ckpt) as f:
            data = json.load(f)
        
        output_name = data.get("output_name", "unknown")
        processed = len(data.get("processed_window_indices", []))
        
        logger.info(f"  {output_name:40s} | Processed: {processed:6d} windows")
    
    logger.info("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Manage training and inference checkpoints"
    )
    
    parser.add_argument(
        "--action",
        type=str,
        choices=["list", "latest", "history", "inference-status"],
        required=True,
        help="Action to perform"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/checkpoints",
        help="Checkpoint directory"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs/logs",
        help="Log directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gis_exports",
        help="Inference output directory"
    )
    
    parser.add_argument(
        "--num-recent",
        type=int,
        default=20,
        help="Number of recent history entries to show"
    )
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_checkpoints(args.checkpoint_dir)
    elif args.action == "latest":
        latest = get_latest_checkpoint(args.checkpoint_dir)
        if latest:
            print(latest)  # Print path for scripting
    elif args.action == "history":
        show_training_history(args.log_dir, args.num_recent)
    elif args.action == "inference-status":
        show_inference_status(args.output_dir)


if __name__ == "__main__":
    main()

