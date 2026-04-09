from src.preprocessing.dataloader import DroneImageDataset
import numpy as np

dataset = DroneImageDataset(
    tiles_dir="data/tiles",
    masks_dir="data/annotations",
    village_names=["Badetumnar"],   # Test one village
    load_to_memory=False
)

print(f"Total tiles loaded: {len(dataset)}")

# Check a few samples
for i in range(5):
    sample = dataset[i]
    print(f"Sample {i}: Image {sample['image'].shape} | Mask {sample['mask'].shape} | "
          f"Unique classes: {np.unique(sample['mask'].numpy())}")