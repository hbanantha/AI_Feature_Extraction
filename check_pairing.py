from pathlib import Path

village = "Badetumnar"  # ← Change if needed

image_dir = Path(f"data/tiles/{village}/tiles")
mask_dir = Path(f"data/annotations/{village}/masks")

image_stems = {f.stem for f in image_dir.glob("*.npy")}
mask_stems = {f.stem for f in mask_dir.glob("*.npy")}

print(f"Image tiles : {len(image_stems)}")
print(f"Mask tiles  : {len(mask_stems)}")

if image_stems == mask_stems:
    print("\n✅ PERFECT MATCH - Filenames are correctly paired!")
    print(f"Total usable tiles = {len(image_stems)}")
else:
    print("\n❌ Filename mismatch detected!")
    missing_mask = image_stems - mask_stems
    missing_image = mask_stems - image_stems

    if missing_mask:
        print(f"   {len(missing_mask)} images have no mask. Example: {list(missing_mask)[:3]}")
    if missing_image:
        print(f"   {len(missing_image)} masks have no image. Example: {list(missing_image)[:3]}")