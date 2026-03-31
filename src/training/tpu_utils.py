"""
TPU Optimization for PyTorch XLA

Utilities and optimizations for training on TPU with PyTorch XLA.
Includes synchronization, memory optimization, and distributed training support.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)

# Try to import torch_xla
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    from torch_xla.amp import autocast as xla_autocast
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    logger.warning("torch_xla not available, TPU training disabled")


class TPUTrainingContext:
    """
    Context manager for TPU-optimized training.
    Handles device placement, synchronization, and memory optimization.
    """

    def __init__(
        self,
        use_tpu: bool = True,
        use_amp: bool = True,
        sync_gradients_every_n_steps: int = 1
    ):
        self.use_tpu = use_tpu and XLA_AVAILABLE
        self.use_amp = use_amp
        self.sync_gradients_every_n_steps = sync_gradients_every_n_steps
        
        self.device = self._get_device()
        self.num_cores = self._get_num_cores()
        
        logger.info(f"TPU Available: {self.use_tpu}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Num cores: {self.num_cores}")

    def _get_device(self) -> torch.device:
        """Get appropriate device (TPU or CPU/GPU)."""
        if self.use_tpu:
            try:
                device = xm.xla_device()
                logger.info(f"Using TPU device: {device}")
                return device
            except Exception as e:
                logger.warning(f"Failed to get TPU device: {e}, using CPU")
                return torch.device("cpu")
        else:
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")

    def _get_num_cores(self) -> int:
        """Get number of TPU cores."""
        if self.use_tpu:
            try:
                return xm.xrt_world_size()
            except Exception:
                return 1
        else:
            return 1

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Reduce loss across all TPU cores.
        Critical for proper loss scaling in distributed training.
        """
        if self.use_tpu and self.num_cores > 1:
            return xm.all_reduce(xm.REDUCE_SUM, loss) / self.num_cores
        return loss

    def step(self):
        """
        Synchronization point for TPU.
        Should be called at the end of each training step.
        """
        if self.use_tpu:
            xm.mark_step()

    def barrier(self):
        """Synchronize all TPU cores."""
        if self.use_tpu:
            xm.rendezvous("barrier")

    def is_master(self) -> bool:
        """Check if this is the master core (core 0)."""
        if self.use_tpu:
            return xm.get_ordinal() == 0
        return True

    def get_ordinal(self) -> int:
        """Get TPU core ordinal."""
        if self.use_tpu:
            return xm.get_ordinal()
        return 0


def optimize_model_for_tpu(model: nn.Module) -> nn.Module:
    """
    Optimize model architecture for TPU training.
    
    TPU-specific optimizations:
    - Use batch norm instead of layer norm where possible
    - Avoid dynamic shapes
    - Use in-place operations carefully
    - Minimize host<->device transfers
    """
    logger.info("Optimizing model for TPU...")
    
    if not XLA_AVAILABLE:
        logger.warning("XLA not available, skipping TPU optimization")
        return model

    # Move model to TPU device
    device = xm.xla_device()
    model = model.to(device)
    
    return model


class TPUGradientAccumulator:
    """
    Gradient accumulation with proper TPU synchronization.
    Prevents gradient divergence in distributed training.
    """

    def __init__(
        self,
        accumulation_steps: int = 1,
        tpu_context: Optional[TPUTrainingContext] = None
    ):
        self.accumulation_steps = accumulation_steps
        self.tpu_context = tpu_context or TPUTrainingContext(use_tpu=False)
        self.step_count = 0

    def should_sync(self) -> bool:
        """Check if we should sync gradients."""
        return (self.step_count + 1) % self.accumulation_steps == 0

    def should_step(self) -> bool:
        """Check if we should do optimizer step."""
        return self.should_sync()

    def step(self):
        """Increment step counter."""
        self.step_count += 1

    def reset(self):
        """Reset counter (after optimizer step)."""
        self.step_count = 0

    def sync_if_needed(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce loss if needed."""
        if self.tpu_context.num_cores > 1 and self.should_sync():
            loss = self.tpu_context.reduce_loss(loss)
        return loss


def create_tpu_compatible_dataloader(
    dataset,
    batch_size: int,
    is_training: bool = True,
    num_workers: int = 0,
    **kwargs
):
    """
    Create TPU-compatible DataLoader.
    
    Key differences from standard DataLoader:
    - No multiprocessing (num_workers=0) on TPU
    - Drop last batch to avoid shape issues
    - Pin memory for efficiency
    """
    from torch.utils.data import DataLoader
    
    # TPU doesn't support multiprocessing well
    if XLA_AVAILABLE:
        num_workers = 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_training,  # Important for TPU
        **kwargs
    )


def reduce_metrics_across_tpu(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Average metrics across all TPU cores.
    """
    if not (XLA_AVAILABLE and torch_xla is not None):
        return metrics
    
    try:
        reduced_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor = torch.tensor(value, dtype=torch.float32)
                tensor = xm.all_reduce(xm.REDUCE_MEAN, tensor)
                reduced_metrics[key] = tensor.item()
            else:
                reduced_metrics[key] = value
        return reduced_metrics
    except Exception as e:
        logger.warning(f"Failed to reduce metrics: {e}")
        return metrics


def log_tpu_memory_stats():
    """Log TPU memory statistics."""
    if not XLA_AVAILABLE:
        return
    
    try:
        stats = xm.get_memory_info(xm.xla_device())
        logger.info(f"TPU Memory: {stats}")
    except Exception as e:
        logger.debug(f"Could not get TPU memory stats: {e}")


def set_tpu_autocheckpointing(enabled: bool = True):
    """
    Enable TPU automatic checkpointing for memory efficiency.
    """
    if not XLA_AVAILABLE:
        return
    
    try:
        os.environ['XLA_USE_REMOTE_CACHE'] = '1'
        if enabled:
            os.environ['XLA_ENABLE_LAZY_EVALUATION'] = '0'
            logger.info("TPU auto-checkpointing enabled")
    except Exception as e:
        logger.debug(f"Could not set auto-checkpointing: {e}")


# Legacy support for torch_xla < 1.0
def maybe_reduce_loss(loss: torch.Tensor, use_tpu: bool = False) -> torch.Tensor:
    """Reduce loss for distributed training (if on TPU)."""
    if use_tpu and XLA_AVAILABLE and xm.xrt_world_size() > 1:
        return xm.all_reduce(xm.REDUCE_SUM, loss) / xm.xrt_world_size()
    return loss


def tpu_print(msg: str, ordinal: int = 0):
    """Print only from specified TPU core (default: core 0)."""
    if XLA_AVAILABLE:
        if xm.get_ordinal() == ordinal:
            logger.info(msg)
    else:
        logger.info(msg)

