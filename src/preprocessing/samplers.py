"""
Sampling strategies for balanced training.

Samplers for handling class imbalance in semantic segmentation.
"""

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from typing import Optional, List, Dict
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class ClassBalancedSampler(Sampler):
    """
    Sampler that balances classes by resampling.
    
    Upsamples minority classes and downsamples majority classes
    to create more balanced batches.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        class_weights: Optional[List[float]] = None,
        class_counts=None
    ):
        """
        Args:
            dataset: Dataset with 'mask' in __getitem__
            num_samples: Number of samples to draw (default: len(dataset))
            replacement: Whether to sample with replacement
            class_weights: Pre-computed weights (if None, computed from data)
        """
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        self.replacement = replacement
        if class_counts is not None:
            self.class_counts = class_counts
        else:
            raise ValueError(
                "class_counts must be provided to avoid recomputation."
            )
        # Compute class frequencies from masks
        if class_weights is None:
            self.class_weights = self._compute_class_weights()
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

        logger.info(f"ClassBalancedSampler initialized with weights: {self.class_weights.tolist()}")

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute sample weights based on class distribution."""
        logger.info("Computing class weights from dataset...")
        
        class_counts = Counter()
        total_pixels = 0

        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                mask = sample.get("mask", None)
                
                if mask is None:
                    continue
                
                # Convert to numpy if needed
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()
                
                # Count classes
                unique, counts = np.unique(mask, return_counts=True)
                for cls, count in zip(unique, counts):
                    class_counts[int(cls)] += count
                    total_pixels += count
                    
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue

        if not class_counts:
            logger.warning("No class counts found, using uniform weights")
            return torch.ones(7)  # Default for 7 classes

        # Compute weights using effective number
        num_classes = max(class_counts.keys()) + 1
        weights = torch.ones(num_classes)
        
        beta = 0.999
        for cls in range(num_classes):
            count = class_counts.get(cls, 1)
            effective_num = 1.0 - np.power(beta, count)
            if effective_num > 0:
                weights[cls] = (1.0 - beta) / effective_num
            else:
                weights[cls] = 1.0

        # Normalize
        weights = weights / weights.sum() * num_classes
        weights = torch.clamp(weights, min=0.1, max=10.0)
        
        logger.info(f"Class distribution: {dict(class_counts)}")
        logger.info(f"Computed weights: {weights.tolist()}")
        
        return weights

    def __iter__(self):
        """Generate sample indices with class balancing."""
        # Create a weight for each sample based on its dominant class
        sample_weights = []
        
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                mask = sample.get("mask", None)
                
                if mask is None:
                    # Default weight
                    sample_weights.append(1.0)
                    continue
                
                # Convert to numpy if needed
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()
                
                # Find dominant class (excluding background if possible)
                unique, counts = np.unique(mask, return_counts=True)
                dominant_cls = unique[np.argmax(counts)]
                
                # Weight sample by its dominant class weight
                if dominant_cls < len(self.class_weights):
                    weight = float(self.class_weights[int(dominant_cls)])
                else:
                    weight = 1.0
                
                sample_weights.append(weight)
                
            except Exception as e:
                logger.warning(f"Error in sampler for index {idx}: {e}")
                sample_weights.append(1.0)

        sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
        
        # Use weighted random sampling
        indices = torch.multinomial(
            sample_weights,
            self.num_samples,
            self.replacement
        )
        
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


class StratifiedSampler(Sampler):
    """
    Stratified sampler that ensures each batch has a mix of classes.
    Better for very imbalanced datasets.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_samples_per_class: Optional[int] = None
    ):
        """
        Args:
            dataset: Dataset instance
            batch_size: Batch size
            num_samples_per_class: Max samples per class (for balancing)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class

        # Group indices by dominant class
        self.class_indices = self._group_by_class()

    def _group_by_class(self) -> Dict[int, List[int]]:
        """Group dataset indices by dominant class."""
        class_indices = {}
        logger.info("Grouping samples by dominant class...")

        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                mask = sample.get("mask", None)
                
                if mask is None:
                    cls = 0
                else:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.numpy()
                    
                    unique, counts = np.unique(mask, return_counts=True)
                    cls = int(unique[np.argmax(counts)])
                
                if cls not in class_indices:
                    class_indices[cls] = []
                class_indices[cls].append(idx)
                
            except Exception as e:
                logger.warning(f"Error grouping sample {idx}: {e}")
                if 0 not in class_indices:
                    class_indices[0] = []
                class_indices[0].append(idx)

        logger.info(f"Class distribution in stratified sampler:")
        for cls, indices in class_indices.items():
            logger.info(f"  Class {cls}: {len(indices)} samples")

        return class_indices

    def __iter__(self):
        """Generate balanced batch indices."""
        num_classes = len(self.class_indices)
        samples_per_class_per_batch = self.batch_size // max(num_classes, 1)
        
        indices = []
        
        # Sample from each class in round-robin fashion
        while True:
            added_any = False
            
            for cls in sorted(self.class_indices.keys()):
                class_samples = self.class_indices[cls]
                
                if len(class_samples) == 0:
                    continue
                
                # Randomly sample from this class
                batch_indices = np.random.choice(
                    class_samples,
                    size=min(samples_per_class_per_batch, len(class_samples)),
                    replace=True
                )
                indices.extend(batch_indices.tolist())
                added_any = True
            
            if not added_any:
                break
            
            # Shuffle within each batch-size chunk
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size:
                    np.random.shuffle(batch)

        return iter(indices[:len(self)])

    def __len__(self):
        return sum(len(v) for v in self.class_indices.values())


# Utility function for creating balanced dataloaders
def create_balanced_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    sampler_type: str = "balanced",
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs
):
    """
    Create a DataLoader with class balancing.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if using custom sampler)
        sampler_type: "balanced" or "stratified"
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        **kwargs: Additional DataLoader arguments
    """
    from torch.utils.data import DataLoader
    
    if sampler_type == "balanced":
        sampler = ClassBalancedSampler(dataset)
    elif sampler_type == "stratified":
        sampler = StratifiedSampler(dataset, batch_size)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),  # Can't use shuffle with custom sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )

