"""
EXAMPLE: How to train with class collapse fixes

This script demonstrates how to use the improved training pipeline
with all the anti-collapse features enabled.
"""

import yaml
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_train_with_fixes():
    """
    Example: Training with all class collapse fixes enabled
    """
    
    # Load config (automatically has all optimizations)
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("TRAINING WITH CLASS COLLAPSE FIXES")
    print("=" * 70)
    
    # OPTION 1: CPU/GPU training
    print("\n[CPU/GPU Training]")
    from src.training import IncrementalTrainer
    
    trainer = IncrementalTrainer(config)
    trainer.setup()
    
    print(f"✓ Model created")
    print(f"✓ Loss function: {type(trainer.loss_fn).__name__}")
    print(f"✓ Optimizer: AdamW")
    print(f"✓ Device: {trainer.device}")
    
    # The trainer automatically:
    # 1. Computes class weights from data
    # 2. Uses ClassBalancedSampler for training
    # 3. Clips gradients (max_norm=1.0)
    # 4. Monitors class diversity per epoch
    # 5. Applies enhanced loss (focal + dice + lovasz)
    
    print("\nAuto-enabled features:")
    print("  ✓ Class-weighted loss")
    print("  ✓ Balanced sampling")
    print("  ✓ Gradient clipping")
    print("  ✓ Class diversity monitoring")
    print("  ✓ Advanced augmentations")
    
    # Train (if you have data)
    # villages = ["village1", "village2"]
    # trainer.train_incremental(villages)


def example_tpu_training():
    """
    Example: TPU training with xla optimization
    """
    
    print("\n" + "=" * 70)
    print("TPU TRAINING WITH XLA OPTIMIZATION")
    print("=" * 70)
    
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        
        print("\n✓ PyTorch XLA detected")
        
        from src.training.tpu_utils import (
            TPUTrainingContext,
            create_tpu_compatible_dataloader
        )
        
        # Create TPU context
        tpu_context = TPUTrainingContext(
            use_tpu=True,
            use_amp=True,  # TPU has excellent AMP support
            sync_gradients_every_n_steps=2
        )
        
        print(f"✓ TPU context initialized")
        print(f"  Device: {tpu_context.device}")
        print(f"  Num cores: {tpu_context.num_cores}")
        print(f"  Is master: {tpu_context.is_master()}")
        
        print("\nTPU-specific optimizations:")
        print("  ✓ Multi-core synchronization")
        print("  ✓ Mixed precision (use_amp=True)")
        print("  ✓ Gradient accumulation (2 steps = effective batch 32)")
        print("  ✓ Automatic loss reduction across cores")
        print("  ✓ Stateless computation for memory efficiency")
        
        print("\nKey settings for TPU:")
        print("  - num_workers: 0 (no multiprocessing)")
        print("  - batch_size: 16 (memory efficient)")
        print("  - drop_last: True (matching shapes)")
        print("  - use_amp: True (better performance)")
        
    except ImportError:
        print("\n⚠ PyTorch XLA not installed")
        print("  Install with: pip install torch-xla[tpu]")
        print("  Or use CPU/GPU training instead")


def example_custom_loss():
    """
    Example: Using the improved loss function directly
    """
    
    print("\n" + "=" * 70)
    print("CUSTOM LOSS CONFIGURATION")
    print("=" * 70)
    
    from src.training.losses import (
        CombinedSegmentationLoss,
        get_class_weights
    )
    
    # Example class distribution (your data)
    class_counts = {
        0: 100000,  # background (common)
        1: 80000,   # building_rcc (common)
        2: 40000,   # building_tiled
        3: 15000,   # building_tin
        4: 20000,   # building_others
        5: 10000,   # road (rare!)
        6: 5000,    # waterbody (rare!)
    }
    
    # Compute class weights using "effective number" method
    class_weights = get_class_weights(class_counts, num_classes=7, method="effective")
    print(f"Computed class weights:\n  {class_weights}")
    
    # Create loss function with all improvements
    loss_fn = CombinedSegmentationLoss(
        ce_weight=0.5,           # Focal cross-entropy weight
        dice_weight=0.4,         # Dice loss weight
        lovasz_weight=0.1,       # Lovasz loss weight (boundary preservation)
        class_weights=class_weights.tolist(),
        use_focal=True,          # Use focal loss instead of standard CE
        use_label_smoothing=True,
        label_smoothing=0.1
    )
    
    print(f"\n✓ Loss function configured:")
    print(f"  - Focal CE: {loss_fn.ce_weight}")
    print(f"  - Weighted Dice: {loss_fn.dice_weight}")
    print(f"  - Lovasz: {loss_fn.lovasz_weight}")
    print(f"  - Label smoothing: {loss_fn.label_smoothing}")
    
    # Test forward pass
    B, C, H, W = 2, 7, 128, 128
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))
    
    loss_dict = loss_fn(pred, target)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  CE loss: {loss_dict['ce']:.4f}")
    print(f"  Dice loss: {loss_dict['dice']:.4f}")
    if 'lovasz' in loss_dict:
        print(f"  Lovasz loss: {loss_dict['lovasz']:.4f}")


def example_balanced_sampling():
    """
    Example: Using class-balanced sampling
    """
    
    print("\n" + "=" * 70)
    print("CLASS-BALANCED SAMPLING")
    print("=" * 70)
    
    from src.preprocessing import ClassBalancedSampler
    from src.preprocessing.dataloader import DroneImageDataset
    
    print("\nBalanced sampling automatically:")
    print("  1. Scans dataset to compute class frequencies")
    print("  2. Assigns higher weight to samples with rare classes")
    print("  3. Uses weighted sampling for each batch")
    print("  4. Result: More balanced class distribution per batch")
    
    print("\nExample batch composition BEFORE:")
    print("  Background: 60% of pixels")
    print("  Buildings: 35% of pixels")
    print("  Roads: 4% of pixels")
    print("  Water: 1% of pixels")
    
    print("\nExample batch composition AFTER:")
    print("  Background: 30% of pixels")
    print("  Buildings: 35% of pixels")
    print("  Roads: 20% of pixels")
    print("  Water: 15% of pixels")
    
    print("\n→ Minority classes get 5-15x more training signal!")


def example_monitoring():
    """
    Example: Monitoring for class collapse
    """
    
    print("\n" + "=" * 70)
    print("CLASS DIVERSITY MONITORING")
    print("=" * 70)
    
    print("\nEvery epoch, the trainer logs:")
    print("  - Per-class prediction counts")
    print("  - Dominant class ratio")
    print("  - Warning if collapse detected (ratio > 95%)")
    
    print("\nExample GOOD output:")
    print("  Epoch class distribution: {0: 50000, 1: 45000, 2: 8000, 3: 3000, 5: 6000, 6: 2000}")
    print("  Dominant class ratio: 0.35 (35% of pixels)")
    
    print("\nExample BAD output (collapse):")
    print("  Epoch class distribution: {0: 110000, 1: 7000, 2: 0, 3: 0, 5: 0, 6: 0}")
    print("  Dominant class ratio: 0.94")
    print("  WARNING: Possible class collapse!")
    
    print("\nIf you see the BAD output, try:")
    print("  1. Increase lovasz_weight to 0.2")
    print("  2. Reduce learning_rate to 0.0001")
    print("  3. Increase gradient_accumulation_steps to 4")


def example_augmentations():
    """
    Example: New satellite-specific augmentations
    """
    
    print("\n" + "=" * 70)
    print("SATELLITE-SPECIFIC AUGMENTATIONS")
    print("=" * 70)
    
    print("\nNew augmentations for satellite imagery:")
    print("  - Perspective transforms: Simulate viewing angle changes")
    print("  - Elastic deformation: Simulate atmospheric distortion")
    print("  - CLAHE: Contrast-limited adaptive histogram equalization")
    print("  - Random rain: Simulate weather effects")
    print("  - Brightness/contrast: Simulate multi-sensor variation")
    
    print("\nWhy for satellite?")
    print("  → Roads/water are thin features (need perspective invariance)")
    print("  → Different satellites have different radiometry (need sensor variation)")
    print("  → Atmospheric effects vary (need robustness to weather)")
    
    print("\nResult: Model learns robust features, not sensor artifacts")


def main():
    """Run all examples"""
    
    print("\n\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "CLASS COLLAPSE FIX - PRACTICAL EXAMPLES".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    example_train_with_fixes()
    example_tpu_training()
    example_custom_loss()
    example_balanced_sampling()
    example_monitoring()
    example_augmentations()
    
    print("\n" + "=" * 70)
    print("QUICK START")
    print("=" * 70)
    print("""
To start training with all fixes enabled:

    python main.py train --config configs/config.yaml

The trainer automatically:
  ✓ Computes class weights from your data
  ✓ Uses balanced sampling for each batch
  ✓ Applies enhanced multi-component loss
  ✓ Clips gradients for stability
  ✓ Monitors class diversity per epoch
  ✓ Uses satellite-specific augmentations

For TPU training:
    python main.py train --config configs/config.yaml --tpu

Key things to watch for:
  1. Check "Epoch class distribution" in logs
  2. Watch for "WARNING: Possible class collapse!" messages
  3. Monitor per-class mIoU (roads and water should improve)
  4. Verify confidence is NOT always 1.0 or 0.0

See CLASS_COLLAPSE_FIX_GUIDE.md for full documentation.
    """)
    
    print("=" * 70)


if __name__ == "__main__":
    main()

