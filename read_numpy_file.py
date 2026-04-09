import numpy as np
import matplotlib.pyplot as plt

tile_path = "data/tiles/Badetumnar/tiles/tile_0010_0029.npy"
mask_path = "data/annotations/Badetumnar/masks/tile_0010_0029.npy"

tile = np.load(tile_path)
mask = np.load(mask_path)

print("Tile shape :", tile.shape)
print("Mask shape :", mask.shape)
print("Unique classes in mask:", np.unique(mask))
print("Background percentage:", (mask == 0).mean() * 100, "%")

plt.figure(figsize=(20, 8))

plt.subplot(1, 3, 1)
plt.imshow(tile)
plt.title("Original Tile")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='tab20')
plt.title("Mask")
plt.axis('off')

# Overlay
plt.subplot(1, 3, 3)
plt.imshow(tile)
plt.imshow(mask, cmap='tab20', alpha=0.5)   # Semi-transparent overlay
plt.title("Tile + Mask Overlay")
plt.axis('off')

plt.tight_layout()
plt.show()