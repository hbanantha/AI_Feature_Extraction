"""
Incremental Training Pipeline
Memory-efficient training with incremental learning support.
"""

import os
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List

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
from src.training.losses import CombinedSegmentationLoss, EWCLoss
from src.training.metrics import SegmentationMetrics

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

        self.loss_fn = CombinedSegmentationLoss(
            ce_weight=0.5,
            dice_weight=0.5,
            use_focal=True
        )

        if self.use_ewc:
            self.ewc_loss = EWCLoss(self.model, lambda_ewc=self.ewc_lambda)

    # --------------------------------------------------------

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

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=is_training,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=is_training
        )

    # --------------------------------------------------------

    def train_epoch(self, loader: DataLoader):

        self.model.train()
        total_loss = 0

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
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss_dict["total"].item()

            if step % 10 == 0:
                self._add_to_replay(batch)

        return total_loss / len(loader)

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
            use_replay=(batch_idx > 0)
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
            val_metrics = self.validate(val_loader)

            self.scheduler.step()

            logger.info(
                f"Epoch {self.current_epoch} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val mIoU: {val_metrics['mIoU']:.4f}"
            )

            if val_metrics["mIoU"] > best_batch_miou:
                best_batch_miou = val_metrics["mIoU"]
                patience_counter = 0
                self.save_checkpoint(f"batch_{batch_idx}_best.pth", val_metrics)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if self.use_ewc:
            self.ewc_loss.compute_fisher(
                self.model,
                train_loader,
                device=self.device
            )

        return best_batch_miou

    # --------------------------------------------------------

    def train_incremental(self, villages: List[str]):

        self.setup()

        val_ratio = 0.2
        n_val = max(1, int(len(villages) * val_ratio))

        val_villages = villages[-n_val:]
        train_villages = villages[:-n_val]

        num_batches = (
            len(train_villages) + self.villages_per_batch - 1
        ) // self.villages_per_batch

        for batch_idx in range(num_batches):

            start = batch_idx * self.villages_per_batch
            end = start + self.villages_per_batch
            batch_villages = train_villages[start:end]

            best_miou = self.train_village_batch(
                batch_villages,
                val_villages,
                batch_idx
            )

            logger.info(
                f"Batch {batch_idx} completed | Best mIoU: {best_miou:.4f}"
            )

        self.save_history()

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

def train(config_path: str):

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
    trainer.train_incremental(villages)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml"
    )

    args = parser.parse_args()
    train(args.config)