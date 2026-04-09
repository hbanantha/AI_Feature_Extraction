"""
Improved Mask Creation - Fixed Transform & CRS Handling
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
import fiona

# ========================= CONFIG =========================
TILES_ROOT   = Path("data/tiles")
ANNOTATIONS_ROOT = Path("data/annotations")
GPKG_DIR     = Path("output_gpkg")
TILE_SIZE    = 256
STRIDE       = 224   # 256 - 32
# =========================================================

def create_masks_for_village(village_name: str):
    print(f"\n🔄 Processing village: {village_name}")

    village_tile_dir = TILES_ROOT / village_name / "tiles"
    metadata_path = TILES_ROOT / village_name / "tiles_metadata.json"
    gpkg_path = GPKG_DIR / f"{village_name}.gpkg"
    mask_dir = ANNOTATIONS_ROOT / village_name / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists() or not gpkg_path.exists():
        print("❌ Metadata or GPKG missing")
        return

    # Load metadata
    with open(metadata_path) as f:
        meta = json.load(f)

    source_tif = meta["source_file"]

    with rasterio.open(source_tif) as src:
        full_transform = src.transform
        src_crs = src.crs

    # Load all features
    all_gdf = []
    for layer in fiona.listlayers(str(gpkg_path)):
        gdf = gpd.read_file(gpkg_path, layer=layer)
        if "class_id" in gdf.columns:
            valid = gdf[gdf["class_id"].notnull()].copy()
            if not valid.empty:
                valid = valid[["geometry", "class_id"]]
                all_gdf.append(valid)

    if not all_gdf:
        print("❌ No valid features")
        return

    merged_gdf = gpd.GeoDataFrame(pd.concat(all_gdf, ignore_index=True))

    # Critical Fix: Reproject to match source TIFF CRS
    if merged_gdf.crs != src_crs:
        print(f"Reprojecting from {merged_gdf.crs} to {src_crs}")
        merged_gdf = merged_gdf.to_crs(src_crs)

    print(f"Total features: {len(merged_gdf)} | Classes: {merged_gdf['class_id'].unique()}")

    # Rasterize each tile
    success_count = 0
    for tile_info in tqdm(meta["tiles"], desc="Creating masks"):
        filename = tile_info["filename"]
        row_idx = tile_info["row_idx"]
        col_idx = tile_info["col_idx"]

        col_off = col_idx * STRIDE
        row_off = row_idx * STRIDE

        window = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)
        tile_transform = rasterio.windows.transform(window, full_transform)

        # Rasterize
        shapes = [(geom, int(cls)) for geom, cls in zip(merged_gdf.geometry, merged_gdf.class_id)
                  if geom is not None and not geom.is_empty]

        mask = rasterize(
            shapes=shapes,
            out_shape=(TILE_SIZE, TILE_SIZE),
            transform=tile_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )

        if np.any(mask != 0):
            success_count += 1

        np.save(mask_dir / filename, mask)

    print(f"✅ Done! {success_count} tiles have features out of {len(meta['tiles'])}")


# ===================== MAIN =====================
if __name__ == "__main__":
    gpkg_files = list(GPKG_DIR.glob("*.gpkg"))
    villages = [f.stem for f in gpkg_files]

    for village in villages:
        create_masks_for_village(village)

    print("\n🎉 Mask creation completed!")