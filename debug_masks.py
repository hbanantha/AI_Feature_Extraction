import numpy as np
from pathlib import Path
from collections import Counter

# ====================== CONFIG ======================
VILLAGE_NAME = "Badetumnar"  # ← CHANGE THIS

TILES_DIR = Path(f"data/tiles/{VILLAGE_NAME}/tiles")
MASKS_DIR = Path(f"data/annotations/{VILLAGE_NAME}/masks")

NUM_TILES_TO_CHECK = 500  # Change to 100 if you want
# ===================================================

print(f"Sampling {NUM_TILES_TO_CHECK} tiles from village: {VILLAGE_NAME}\n")

all_classes = []
non_bg_tiles = 0
total_non_bg_pixels = 0

mask_files = sorted(MASKS_DIR.glob("*.npy"))[:NUM_TILES_TO_CHECK]

for mask_file in mask_files:
    tile_file = TILES_DIR / mask_file.name

    mask = np.load(mask_file)

    unique = np.unique(mask)
    non_bg = np.sum(mask != 0)

    print(f"{mask_file.name:30} | Classes: {unique} | Non-bg pixels: {non_bg:6,d}")

    all_classes.extend(unique)
    if non_bg > 0:
        non_bg_tiles += 1
        total_non_bg_pixels += non_bg

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Tiles checked          : {len(mask_files)}")
print(f"Tiles with features    : {non_bg_tiles} ({non_bg_tiles / len(mask_files) * 100:.1f}%)")
print(f"Total non-bg pixels    : {total_non_bg_pixels:,}")
print(f"Overall classes found  : {sorted(set(all_classes))}")
print(f"Class distribution     : {Counter(all_classes)}")