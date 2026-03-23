from src.preprocessing import DroneImageDataset, get_training_augmentation
import yaml

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

transform = get_training_augmentation(config)
dataset = DroneImageDataset(
    tiles_dir=config['data']['tiles_dir'],
    masks_dir=config['data']['annotations_dir'],
    transform=transform,
    is_training=True,
    village_names=['BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO']
)

print(f'Dataset size: {len(dataset)}')
sample = dataset[0]
print(f'Image shape: {sample["image"].shape}, dtype: {sample["image"].dtype}')
print(f'Mask shape: {sample["mask"].shape}, dtype: {sample["mask"].dtype}')
print(f'Mask min: {sample["mask"].min()}, max: {sample["mask"].max()}')
print('Test passed!')
