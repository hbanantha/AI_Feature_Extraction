"""
Debug script to check if dataloader can find tiles and masks
"""
from pathlib import Path
from src.preprocessing.dataloader import DroneImageDataset
import yaml

# Load config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Create dataset
dataset = DroneImageDataset(
    tiles_dir=config["data"]["tiles_dir"],
    masks_dir=config["data"]["annotations_dir"],
    village_names=None,
    is_training=True,
    transform=None
)

print(f"Total tiles found: {len(dataset.tile_paths)}")
print(f"Total masks found: {len(dataset.mask_paths)}")

if dataset.tile_paths:
    print(f"\nFirst 5 tiles:")
    for i, tile_path in enumerate(dataset.tile_paths[:5]):
        print(f"  {i+1}. {tile_path}")

if dataset.mask_paths:
    print(f"\nFirst 5 masks:")
    for i, mask_path in enumerate(dataset.mask_paths[:5]):
        mask_exists = "✓" if mask_path and Path(mask_path).exists() else "✗"
        print(f"  {i+1}. {mask_path} {mask_exists}")
