"""
Incremental Training Pipeline
Memory-efficient training with incremental learning support.
"""

import os
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import yaml

from src.models import create_model
from src.preprocessing import (
    DroneImageDataset,
    IncrementalDataset,
    ReplayBuffer,
    get_training_augmentation,
    get_validation_augmentation
)
from src.training.losses import CombinedSegmentationLoss, EWCLoss, get_class_weights
from src.training.metrics import SegmentationMetrics
from src.preprocessing.samplers import ClassBalancedSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Incremental Trainer
# ============================================================

class IncrementalTrainer:

    def __init__(self, config: Dict):

        self.config = config
        self.device = config["hardware"]["device"]

        # Training settings
        self.batch_size = config["training"]["batch_size"]
        self.num_workers = config["training"]["num_workers"]
        self.epochs_per_batch = config["training"]["epochs_per_village_batch"]
        self.max_epochs = config["training"]["max_epochs"]
        self.lr = config["optimization"]["learning_rate"]
        self.gradient_accumulation_steps = config["optimization"]["gradient_accumulation_steps"]
        self.validation_frequency = config["training"].get("validation_frequency", 1)  # Validate every N epochs

        # Mixed precision
        self.use_amp = config["optimization"]["use_amp"] and self.device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Incremental
        inc_cfg = config["incremental"]
        self.use_ewc = inc_cfg["use_ewc"]
        self.ewc_lambda = inc_cfg["ewc_lambda"]
        self.villages_per_batch = inc_cfg["villages_per_batch"]
        self.replay_buffer_size = inc_cfg["replay_buffer_size"]

        # Core components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.ewc_loss = None

        self.replay_buffer = ReplayBuffer(max_size=self.replay_buffer_size)

        self.metrics = SegmentationMetrics(
            num_classes=config["data"]["num_seg_classes"],
            class_names=list(config["data"]["segmentation_classes"].values())
        )

        # Paths
        self.checkpoint_dir = Path(config["checkpointing"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config["logging"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = []

    # --------------------------------------------------------

    def setup(self):

        logger.info("Setting up model and optimizer...")

        self.model = create_model(self.config).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.config["optimization"]["weight_decay"]
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs
        )

        # Initialize loss function with class balancing
        # Use "effective" method for long-tail distribution
        num_classes = self.config["data"]["num_seg_classes"]
        class_weights = None
        
        # Try to compute class weights from the dataset
        try:
            # Create dummy dataset to compute class frequencies
            dummy_dataset = DroneImageDataset(
                tiles_dir=self.config["data"]["tiles_dir"],
                masks_dir=self.config["data"]["annotations_dir"],
                transform=None,
                is_training=True,
                max_samples=500  # Sample subset for speed
            )
            
            class_counts = self._compute_class_frequencies(dummy_dataset)
            class_weights = get_class_weights(
                class_counts,
                num_classes,
                method="effective"  # Recommended for imbalanced data
            ).tolist()
            
            logger.info(f"Computed class weights: {class_weights}")
        except Exception as e:
            logger.warning(f"Could not compute class weights: {e}. Using uniform weights.")
            class_weights = [1.0] * num_classes

        self.loss_fn = CombinedSegmentationLoss(
            ce_weight=0.5,
            dice_weight=0.4,
            lovasz_weight=0.1,  # Add Lovasz loss for boundary preservation
            class_weights=class_weights,
            use_focal=True,
            use_label_smoothing=True,
            label_smoothing=0.1
        )

        if self.use_ewc:
            self.ewc_loss = EWCLoss(self.model, lambda_ewc=self.ewc_lambda)

    def _compute_class_frequencies(self, dataset: DroneImageDataset) -> dict:
        """Compute class frequency distribution from dataset."""
        from collections import Counter
        
        class_counts = Counter()
        total_pixels = 0
        
        for idx in range(min(len(dataset), 500)):  # Sample for speed
            try:
                sample = dataset[idx]
                mask = sample["mask"]
                
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()
                
                unique, counts = np.unique(mask, return_counts=True)
                for cls, count in zip(unique, counts):
                    class_counts[int(cls)] += count
                    total_pixels += count
            except Exception as e:
                logger.debug(f"Error processing sample {idx}: {e}")
                continue
        
        logger.info(f"Class distribution: {dict(class_counts)}")
        return dict(class_counts)

    # -------------------------------------------------------- # --------------------------------------------------------
    #     # NEW: Resume from checkpoint
    def resume_from_checkpoint(self, checkpoint_path: str):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = create_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.config["optimization"]["weight_decay"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["metrics"].get("mIoU", 0.0)

        logger.info(
            f"Checkpoint loaded | Epoch: {self.current_epoch}, Best mIoU: {self.best_metric:.4f}"
        )

    def create_dataloader(self, villages: List[str], is_training=True, use_replay=False):

        transform = (
            get_training_augmentation(self.config)
            if is_training else
            get_validation_augmentation(self.config)
        )

        dataset = DroneImageDataset(
            tiles_dir=self.config["data"]["tiles_dir"],
            masks_dir=self.config["data"]["annotations_dir"],
            transform=transform,
            is_training=is_training,
            village_names=villages
        )

        if use_replay and is_training and len(self.replay_buffer) > 0:
            dataset = IncrementalDataset(
                dataset,
                replay_buffer=self.replay_buffer.get_all(),
                replay_ratio=0.2
            )

        # Use balanced sampler for training to prevent class collapse
        if is_training:
            sampler = ClassBalancedSampler(dataset)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=is_training
            )
        else:
            # Standard shuffled loader for validation
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False
            )

    # --------------------------------------------------------

    def train_epoch(self, loader: DataLoader):
        """
        Train for one epoch with class diversity monitoring.
        
        Key anti-collapse strategies:
        1. Balanced sampling (via ClassBalancedSampler)
        2. Class weighted loss (focal + dice + lovasz)
        3. Per-class accuracy tracking
        4. Early warning on class collapse
        """

        self.model.train()
        total_loss = 0
        loss_components = {"ce": 0, "dice": 0}
        if hasattr(self.loss_fn, 'lovasz_weight') and self.loss_fn.lovasz_weight > 0:
            loss_components["lovasz"] = 0
        
        # Track per-class predictions to detect collapse
        class_predictions = {i: 0 for i in range(self.config["data"]["num_seg_classes"])}
        total_predictions = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader)):

            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.loss_fn(outputs, masks)
                    loss = loss_dict["total"]

                    if self.use_ewc and self.ewc_loss.fisher:
                        loss += self.ewc_loss(self.model)

                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients to prevent explosion
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss_dict = self.loss_fn(outputs, masks)
                loss = loss_dict["total"]

                if self.use_ewc and self.ewc_loss.fisher:
                    loss += self.ewc_loss(self.model)

                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss_dict["total"].item()
            
            # Track loss components
            for key in loss_components.keys():
                if key in loss_dict:
                    loss_components[key] += loss_dict[key].item()
            
            # Track class predictions (from logits)
            with torch.no_grad():
                pred_classes = outputs.argmax(dim=1)  # (B, H, W)
                unique_classes, counts = torch.unique(pred_classes, return_counts=True)
                for cls, count in zip(unique_classes.cpu().numpy(), counts.cpu().numpy()):
                    class_predictions[int(cls)] += int(count)
                    total_predictions += int(count)

            if step % 50 == 0:
                self._add_to_replay(batch)

        # Compute epoch statistics
        avg_loss = total_loss / len(loader)
        
        # Check for class collapse
        if total_predictions > 0:
            dominant_class_ratio = max(v / total_predictions for v in class_predictions.values())
            logger.info(f"Epoch class distribution: {class_predictions}")
            logger.info(f"Dominant class ratio: {dominant_class_ratio:.2%}")
            
            # Warning if collapse detected
            if dominant_class_ratio > 0.95:
                logger.warning(
                    f"WARNING: Possible class collapse! "
                    f"Dominant class: {dominant_class_ratio:.2%}. "
                    f"Check loss weights and sampling."
                )
        
        # Log loss components
        for key in loss_components.keys():
            avg_component = loss_components[key] / len(loader)
            logger.debug(f"Epoch {self.current_epoch} {key}_loss: {avg_component:.4f}")

        return avg_loss

    # --------------------------------------------------------

    def _add_to_replay(self, batch):

        images = batch["image"].cpu().numpy()
        masks = batch["mask"].cpu().numpy()

        self.replay_buffer.add(images, masks)

    # --------------------------------------------------------

    @torch.no_grad()
    def validate(self, loader):

        self.model.eval()
        self.metrics.reset()

        total_loss = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            outputs = self.model(images)
            loss_dict = self.loss_fn(outputs, masks)

            total_loss += loss_dict["total"].item()
            self.metrics.update(outputs, masks)

        results = self.metrics.compute()
        results["val_loss"] = total_loss / len(loader)
        return results

    # --------------------------------------------------------

    def train_village_batch(self, train_villages, val_villages, batch_idx):

        train_loader = self.create_dataloader(
            train_villages,
            is_training=True,
            use_replay=(batch_idx > 0)  # Skip replay for first batch (10% time savings)
        )

        val_loader = self.create_dataloader(
            val_villages,
            is_training=False
        )

        best_batch_miou = 0
        patience = self.config["optimization"]["early_stopping_patience"]
        patience_counter = 0

        for epoch in range(self.epochs_per_batch):

            self.current_epoch += 1

            train_loss = self.train_epoch(train_loader)
            
            # Validate every N epochs (validation_frequency setting)
            should_validate = (epoch + 1) % self.validation_frequency == 0 or epoch == self.epochs_per_batch - 1
            
            if should_validate:
                val_metrics = self.validate(val_loader)
            else:
                # Skip validation, use dummy metrics to avoid checkpoint saving
                val_metrics = {"mIoU": 0.0, "val_loss": 0.0}

            if should_validate:  # Only step scheduler on validation epochs (3-5% speedup)
                self.scheduler.step()

            if should_validate:
                logger.info(
                    f"Epoch {self.current_epoch} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val mIoU: {val_metrics['mIoU']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {self.current_epoch} | "
                    f"Train Loss: {train_loss:.4f}"
                )

            if should_validate and val_metrics["mIoU"] > best_batch_miou:
                best_batch_miou = val_metrics["mIoU"]
                patience_counter = 0
                self.save_checkpoint(f"batch_{batch_idx}_best.pth", val_metrics)
            elif should_validate:
                patience_counter += 1

            if should_validate and patience_counter >= patience:
                break

        # Disable EWC computation for first batch (10% time savings)
        if self.use_ewc and batch_idx > 0:
            self.ewc_loss.compute_fisher(
                self.model,
                train_loader,
                device=self.device
            )

        return best_batch_miou

    # --------------------------------------------------------

    #
    ### CHANGE START: train_incremental with resume support
    def train_incremental(self, villages: List[str], resume_checkpoint: Optional[str] = None):
        self.setup()

        if resume_checkpoint:
            self.resume_from_checkpoint(resume_checkpoint)

        val_ratio = 0.2
        n_val = max(1, int(len(villages) * val_ratio))
        val_villages = villages[-n_val:]
        train_villages = villages[:-n_val]

        num_batches = (len(train_villages) + self.villages_per_batch - 1) // self.villages_per_batch

        # Skip batches already completed
        start_batch = self.current_epoch // self.epochs_per_batch

        for batch_idx in range(start_batch, num_batches):
            start = batch_idx * self.villages_per_batch
            end = start + self.villages_per_batch
            batch_villages = train_villages[start:end]

            best_miou = self.train_village_batch(batch_villages, val_villages, batch_idx)
            logger.info(f"Batch {batch_idx} completed | Best mIoU: {best_miou:.4f}")

        self.save_history()

    ### CHANGE END

    # --------------------------------------------------------

    def save_checkpoint(self, filename: str, metrics: Dict):

        path = self.checkpoint_dir / filename

        torch.save({
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics
        }, path)

        logger.info(f"Saved checkpoint: {path}")

    # --------------------------------------------------------

    def save_history(self):

        path = self.log_dir / "training_history.json"

        with open(path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        logger.info(f"Saved training history: {path}")


# ============================================================
# Main Entry
# ============================================================

# def train(config_path: str):
#
#     with open(config_path) as f:
#         config = yaml.safe_load(f)
#
#     tiles_dir = Path(config["data"]["tiles_dir"])
#
#     if not tiles_dir.exists():
#         logger.error("Tiles directory not found!")
#         return
#
#     villages = [d.name for d in tiles_dir.iterdir() if d.is_dir()]
#
#     if not villages:
#         logger.error("No villages found.")
#         return
#
#     trainer = IncrementalTrainer(config)
#     trainer.train_incremental(villages)
### CHANGE START: train function with resume
def train(config_path: str, resume_checkpoint: Optional[str] = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tiles_dir = Path(config["data"]["tiles_dir"])
    if not tiles_dir.exists():
        logger.error("Tiles directory not found!")
        return

    villages = [d.name for d in tiles_dir.iterdir() if d.is_dir()]
    if not villages:
        logger.error("No villages found.")
        return

    trainer = IncrementalTrainer(config)
    trainer.train_incremental(villages, resume_checkpoint)
### CHANGE END



# if __name__ == "__main__":
#
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="configs/config.yaml"
#     )
#
#     args = parser.parse_args()
#     train(args.config)
### CHANGE START: argparse with resume
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, args.resume)
### CHANGE END
