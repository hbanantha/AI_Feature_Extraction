# Quick Reference: Training & Inference with Optimizations

## 🚀 Training

### Start New Training
```bash
python main.py train --config configs/config.yaml
```

### Resume Training from Last Checkpoint
```bash
python main.py train --config configs/config.yaml \
    --resume $(python scripts/manage_training.py --action latest)
```

### Resume from Specific Checkpoint
```bash
python main.py train --config configs/config.yaml \
    --resume outputs/checkpoints/batch_0_best.pth
```

## 🔍 Inference

### Process Single GeoTIFF (Auto-Resume if Interrupted)
```bash
python main.py inference \
    --config configs/config.yaml \
    --model outputs/checkpoints/batch_0_best.pth \
    --input data/test/image.tif
```

### Process Directory of Images
```bash
python main.py inference \
    --config configs/config.yaml \
    --model outputs/checkpoints/batch_0_best.pth \
    --input data/test/  \
    --output outputs/predictions
```

## 📊 Monitoring & Management

### List All Checkpoints
```bash
python scripts/manage_training.py --action list
```

**Output**:
```
Found 5 checkpoints:
  batch_0_best.pth                    | Size:  45.23MB | Modified: 2026-04-12 15:32:45
  batch_1_best.pth                    | Size:  45.23MB | Modified: 2026-04-12 16:05:12
  ...
```

### Show Training Progress (Last 20 Epochs)
```bash
python scripts/manage_training.py --action history
```

**Output**:
```
Training History (showing last 20 entries):
Epoch   Batch   Train Loss   Val Loss       mIoU
    1       0        0.4321       0.3912    0.6543
    2       0        0.3891       0.3456    0.7123
    ...
    Best mIoU: 0.8234 (Epoch 18)
```

### Show All Recent Training (Last N Epochs)
```bash
python scripts/manage_training.py --action history --num-recent 50
```

### Check Incomplete Inference Jobs
```bash
python scripts/manage_training.py --action inference-status
```

**Output**:
```
Found 2 incomplete inference jobs:
  large_image_1                        | Processed: 1250 windows
  large_image_2                        | Processed:  856 windows
```

### Get Latest Checkpoint Path (for Scripting)
```bash
CHECKPOINT=$(python scripts/manage_training.py --action latest)
echo $CHECKPOINT
# Output: outputs/checkpoints/batch_0_best.pth
```

## ⚙️ Configuration Tuning

### For Faster Training
Edit `configs/config.yaml`:
```yaml
training:
  validation_frequency: 2         # Validate every 2 epochs instead of 1
  epochs_per_village_batch: 2     # Fewer epochs per village
  
incremental:
  villages_per_batch: 2           # Process 2 villages per batch
```

### For Better Accuracy
Edit `configs/config.yaml`:
```yaml
training:
  epochs_per_village_batch: 5     # More epochs per village
  validation_frequency: 1         # Validate every epoch
  
augmentation:
  train:
    horizontal_flip: 0.7          # Increase augmentation
    vertical_flip: 0.7
    rotate_90: 0.7
    random_brightness_contrast: 0.5
    gaussian_noise: 0.2
```

### For Faster Inference
Edit `configs/config.yaml`:
```yaml
inference:
  stride: 256                     # Larger stride = fewer windows (faster)
  batch_size: 4                   # Larger batches if GPU available
```

### For Larger Batches (GPU Only)
Edit `configs/config.yaml`:
```yaml
training:
  batch_size: 16                  # Increase from 8
  num_workers: 8                  # More workers if CPU has cores
  
optimization:
  use_amp: true                   # Enable mixed precision on GPU
```

## 🔄 Resumption Examples

### Scenario 1: Training Interrupted
```bash
# Restart the same command - it will resume automatically
python main.py train --config configs/config.yaml --resume outputs/checkpoints/batch_0_best.pth
```

### Scenario 2: Inference Interrupted
```bash
# Restart the same inference command
# It will automatically detect and resume from checkpoint
python main.py inference \
    --config configs/config.yaml \
    --model outputs/checkpoints/batch_0_best.pth \
    --input data/test/large_image.tif

# Shows progress: "Resuming from window 1250/5000"
```

### Scenario 3: Check What Was Done
```bash
# Before resuming, check status
python scripts/manage_training.py --action inference-status

# Output: Shows which images are incomplete and how many windows processed
```

## 📈 Performance Tips

### Tip 1: Monitor Training in Real-Time
```bash
# In terminal 1 - Start training
python main.py train --config configs/config.yaml

# In terminal 2 - Monitor progress (every 5 seconds)
watch -n 5 'python scripts/manage_training.py --action history'
```

### Tip 2: Batch Process Multiple Images
```bash
# Create a script to process all images sequentially with resumption

for image in data/test/*.tif; do
    echo "Processing: $image"
    python main.py inference \
        --config configs/config.yaml \
        --model outputs/checkpoints/batch_0_best.pth \
        --input "$image"
    
    # Check status
    python scripts/manage_training.py --action inference-status
done
```

### Tip 3: Find Best Checkpoint Before Inference
```bash
# Get the checkpoint with best mIoU
BEST_CHECKPOINT=$(python scripts/manage_training.py --action latest)

echo "Using checkpoint: $BEST_CHECKPOINT"

# Show training history to verify
python scripts/manage_training.py --action history --num-recent 5

# Run inference
python main.py inference \
    --config configs/config.yaml \
    --model $BEST_CHECKPOINT \
    --input data/test/
```

## 🛠️ Troubleshooting

**Q: Training stopped unexpectedly, how do I resume?**  
A: Run the same command with `--resume outputs/checkpoints/batch_0_best.pth`. It will continue from that epoch.

**Q: Inference stopped mid-way, will it restart from beginning?**  
A: No! It automatically resumes from checkpoint. Just run the same command again.

**Q: How do I know training is making progress?**  
A: Run `python scripts/manage_training.py --action history` to see mIoU improving.

**Q: How do I check if large inference job is complete?**  
A: Run `python scripts/manage_training.py --action inference-status`. If not listed, it's complete!

**Q: Can I use a different checkpoint for inference?**  
A: Yes! Pass `--model path/to/checkpoint.pth` to use any saved checkpoint.

## 📝 Performance Baseline

With optimizations applied:

| Task | Time | Status |
|---|---|---|
| Trainer setup | ~20 seconds | ⚡ -40% from original |
| Per-epoch training | ~4-5 minutes | ⚡ -10% from original |
| Inference (256x256 image) | ~30 seconds | ⚡ -8% from original |
| Inference (2000x2000 image) | ~45 minutes | ✅ Resumable |

## 🎯 Next Steps

1. **Start Training**: `python main.py train --config configs/config.yaml`
2. **Monitor Progress**: `python scripts/manage_training.py --action history`
3. **Run Inference**: `python main.py inference --config configs/config.yaml --model outputs/checkpoints/batch_0_best.pth --input data/test/`
4. **Check Results**: Outputs in `outputs/gis_exports/`

---

For detailed information, see `OPTIMIZATION_SUMMARY.md`

