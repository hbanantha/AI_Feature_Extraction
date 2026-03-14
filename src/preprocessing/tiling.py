"""
GeoTIFF Tiling Pipeline for Large Drone Images
==============================================
Memory-efficient processing of large GeoTIFF files (>6.5GB)
using windowed reading with rasterio.
Optimized for systems with limited RAM (12GB).
"""

import os
import gc
import json
import logging
from pathlib import Path
from typing import Tuple, Generator, Optional, Dict, List

import numpy as np
import psutil
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# GeoTIFF Tiler
# =============================================================================
class GeoTIFFTiler:
    """
    Memory-efficient tiling of large GeoTIFF files.
    Uses windowed reading to process files larger than available RAM.
    """

    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 32,
        min_valid_ratio: float = 0.7,
        output_format: str = "npy",
        memory_limit_gb: float = 8.0,
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_valid_ratio = min_valid_ratio
        self.output_format = output_format
        self.memory_limit_gb = memory_limit_gb
        self.stride = tile_size - overlap

    # -------------------------------------------------------------------------
    # Memory Utilities
    # -------------------------------------------------------------------------
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)

    def check_memory(self):
        """Check if memory usage exceeds limit."""
        current_usage = self.get_memory_usage()
        if current_usage > self.memory_limit_gb:
            logger.warning(
                f"Memory usage ({current_usage:.2f} GB) exceeds limit. Running GC..."
            )
            gc.collect()

    # -------------------------------------------------------------------------
    # Window Generator
    # -------------------------------------------------------------------------
    def get_tile_windows(
        self, width: int, height: int
    ) -> Generator[Tuple[int, int, Window], None, None]:
        """Generate sliding windows for tiling."""

        row_idx = 0
        for row_off in range(0, height - self.tile_size + 1, self.stride):
            col_idx = 0
            for col_off in range(0, width - self.tile_size + 1, self.stride):
                window = Window(col_off, row_off, self.tile_size, self.tile_size)
                yield row_idx, col_idx, window
                col_idx += 1
            row_idx += 1

    # -------------------------------------------------------------------------
    # Tile Validation
    # -------------------------------------------------------------------------
    def is_valid_tile(self, tile: np.ndarray) -> bool:
        """Check if tile has sufficient valid (non-zero, non-NaN) pixels."""
        if tile.dtype in [np.float32, np.float64]:
            valid_mask = (~np.isnan(tile)) & (tile != 0)
        else:
            valid_mask = tile != 0

        valid_ratio = np.mean(valid_mask)
        return valid_ratio >= self.min_valid_ratio

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------
    def normalize_tile(self, tile: np.ndarray) -> np.ndarray:
        """Normalize tile to uint8 (0–255)."""

        if tile.dtype == np.uint8:
            return tile

        if tile.dtype == np.uint16:
            tile = (tile / 256).astype(np.uint8)

        elif tile.dtype in [np.float32, np.float64]:
            if tile.max() <= 1.0:
                tile = (tile * 255).astype(np.uint8)
            else:
                tile = np.clip(tile, 0, 255).astype(np.uint8)

        else:
            tile = tile.astype(np.uint8)

        return tile

    # -------------------------------------------------------------------------
    # Main Processing Function
    # -------------------------------------------------------------------------
    def process_geotiff(
        self,
        input_path: str,
        output_dir: str,
        village_name: Optional[str] = None,
    ) -> Dict:

        input_path = Path(input_path)
        output_dir = Path(output_dir)

        if village_name is None:
            village_name = input_path.stem

        tiles_output_dir = output_dir / village_name / "tiles"
        tiles_output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "source_file": str(input_path),
            "village_name": village_name,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "stride": self.stride,
            "tiles": [],
        }

        logger.info(f"Processing: {input_path}")

        with rasterio.open(input_path) as src:
            width, height = src.width, src.height
            crs = src.crs
            transform = src.transform

            metadata.update(
                {
                    "source_width": width,
                    "source_height": height,
                    "crs": str(crs) if crs else None,
                    "bounds": list(src.bounds),
                }
            )

            valid_tiles = 0
            skipped_tiles = 0

            for row_idx, col_idx, window in tqdm(
                self.get_tile_windows(width, height),
                desc="Extracting tiles",
            ):
                self.check_memory()

                try:
                    tile = src.read(window=window)
                    tile = np.transpose(tile, (1, 2, 0))

                    if tile.shape[2] > 3:
                        tile = tile[:, :, :3]

                    if not self.is_valid_tile(tile):
                        skipped_tiles += 1
                        continue

                    tile = self.normalize_tile(tile)

                    tile_name = f"tile_{row_idx:04d}_{col_idx:04d}"

                    if self.output_format == "npy":
                        tile_path = tiles_output_dir / f"{tile_name}.npy"
                        np.save(tile_path, tile)
                    else:
                        tile_path = tiles_output_dir / f"{tile_name}.tif"
                        tile_transform = rasterio.windows.transform(
                            window, transform
                        )
                        self._save_geotiff_tile(
                            tile, tile_path, tile_transform, crs
                        )

                    metadata["tiles"].append(
                        {
                            "filename": tile_path.name,
                            "row_idx": row_idx,
                            "col_idx": col_idx,
                        }
                    )

                    valid_tiles += 1

                except Exception as e:
                    logger.error(
                        f"Error processing tile ({row_idx}, {col_idx}): {e}"
                    )

        metadata["total_valid_tiles"] = valid_tiles
        metadata["total_skipped_tiles"] = skipped_tiles

        metadata_path = output_dir / village_name / "tiles_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Completed: {valid_tiles} valid tiles, {skipped_tiles} skipped"
        )

        return metadata

    # -------------------------------------------------------------------------
    # Save GeoTIFF Tile
    # -------------------------------------------------------------------------
    def _save_geotiff_tile(
        self,
        tile: np.ndarray,
        output_path: Path,
        transform,
        crs,
    ):
        tile_chw = np.transpose(tile, (2, 0, 1))

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=tile.shape[0],
            width=tile.shape[1],
            count=tile.shape[2],
            dtype=tile.dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(tile_chw)


# =============================================================================
# Batch Processor
# =============================================================================
class BatchTileProcessor:
    """Process multiple GeoTIFF files in batches."""

    def __init__(self, tiler: GeoTIFFTiler, villages_per_batch: int = 2):
        self.tiler = tiler
        self.villages_per_batch = villages_per_batch

    def process_all_villages(
        self,
        input_dir: str,
        output_dir: str,
    ) -> List[Dict]:

        input_dir = Path(input_dir)
        tiff_files = list(input_dir.glob("*.tif")) + list(
            input_dir.glob("*.tiff")
        )

        all_batches = []

        for batch_idx in range(
            0, len(tiff_files), self.villages_per_batch
        ):
            batch_files = tiff_files[
                batch_idx : batch_idx + self.villages_per_batch
            ]

            batch_meta = []
            for file in batch_files:
                meta = self.tiler.process_geotiff(
                    str(file), output_dir
                )
                batch_meta.append(meta)
                gc.collect()

            all_batches.append(batch_meta)

        return all_batches


# =============================================================================
# Main
# =============================================================================
def main():
    """Main execution entry."""

    tiler = GeoTIFFTiler(
        tile_size=256,
        overlap=32,
        min_valid_ratio=0.7,
        output_format="npy",
        memory_limit_gb=8.0,
    )

    print("GeoTIFF Tiler initialized successfully!")
    print(f"Tile size: {tiler.tile_size}")
    print(f"Overlap: {tiler.overlap}")
    print(f"Memory limit: {tiler.memory_limit_gb} GB")


if __name__ == "__main__":
    main()