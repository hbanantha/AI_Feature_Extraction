"""
Inference Pipeline for Feature Extraction
==========================================
Sliding window inference on large GeoTIFF files
with prediction stitching and post-processing.
"""

import os
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
import cv2

from ..models import load_model, create_model
from ..preprocessing import get_validation_augmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract features from large drone images using sliding window inference.
    Outputs georeferenced predictions as GeoTIFF and Shapefile.
    """

    def __init__(
        self,
        config: Dict,
        model_path: str,
        device: str = "cpu"
    ):
        """
        Initialize feature extractor.

        Args:
            config: Configuration dictionary
            model_path: Path to trained model checkpoint
            device: Device to use for inference
        """
        self.config = config
        self.device = device

        # Inference settings
        self.tile_size = config["data"]["tile_size"]
        self.stride = config["inference"]["stride"]
        self.batch_size = config["inference"]["batch_size"]
        self.confidence_threshold = config["inference"]["confidence_threshold"]

        # Class information
        self.num_classes = config["data"]["num_seg_classes"]
        self.class_names = list(config["data"]["segmentation_classes"].values())
        self.class_colors = self._get_class_colors()

        # Load model
        self.model = load_model(model_path, config, device)
        self.model.eval()

        # Preprocessing
        self.transform = get_validation_augmentation(config)

        # Output settings
        self.output_dir = Path(config["inference"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Feature extractor initialized")
        logger.info(f"Tile size: {self.tile_size}, Stride: {self.stride}")

    def _get_class_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """Get color mapping for visualization."""
        return {
            0: (0, 0, 0),        # Background - Black
            1: (255, 0, 0),      # Building RCC - Red
            2: (0, 255, 0),      # Building Tiled - Green
            3: (0, 0, 255),      # Building Tin - Blue
            4: (255, 255, 0),    # Building Others - Yellow
            5: (128, 128, 128),  # Road - Gray
            6: (0, 255, 255),    # Water body - Cyan
        }

    def extract_features(
        self,
        input_path: str,
        output_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Extract features from a large GeoTIFF file.

        Args:
            input_path: Path to input GeoTIFF
            output_name: Name for output files

        Returns:
            Dictionary with output file paths
        """
        input_path = Path(input_path)

        if output_name is None:
            output_name = input_path.stem

        logger.info(f"Processing: {input_path}")

        with rasterio.open(input_path) as src:
            # Get image properties
            width = src.width
            height = src.height
            crs = src.crs
            transform = src.transform

            logger.info(f"Image size: {width} x {height}")
            logger.info(f"CRS: {crs}")

            # Initialize prediction accumulator
            prediction_sum = np.zeros((self.num_classes, height, width), dtype=np.float32)
            prediction_count = np.zeros((height, width), dtype=np.float32)

            # Generate windows
            windows = self._generate_windows(width, height)
            logger.info(f"Total windows: {len(windows)}")

            # Process in batches
            batch_tiles = []
            batch_windows = []

            for window in tqdm(windows, desc="Extracting features"):
                # Read tile
                tile = src.read(window=window)
                tile = np.transpose(tile, (1, 2, 0))  # CHW -> HWC

                # Handle different channel counts
                if tile.shape[2] > 3:
                    tile = tile[:, :, :3]
                elif tile.shape[2] < 3:
                    tile = np.stack([tile[:, :, 0]] * 3, axis=2)

                # Normalize to 0-255 uint8
                if tile.dtype != np.uint8:
                    if tile.max() > 1:
                        tile = np.clip(tile, 0, 255).astype(np.uint8)
                    else:
                        tile = (tile * 255).astype(np.uint8)

                batch_tiles.append(tile)
                batch_windows.append(window)

                # Process batch
                if len(batch_tiles) >= self.batch_size:
                    self._process_batch(
                        batch_tiles, batch_windows,
                        prediction_sum, prediction_count
                    )
                    batch_tiles = []
                    batch_windows = []

                    # Memory cleanup
                    gc.collect()

            # Process remaining tiles
            if batch_tiles:
                self._process_batch(
                    batch_tiles, batch_windows,
                    prediction_sum, prediction_count
                )

        # Average predictions
        prediction_count = np.maximum(prediction_count, 1)  # Avoid division by zero
        prediction_avg = prediction_sum / prediction_count

        # Get final class predictions
        class_predictions = np.argmax(prediction_avg, axis=0).astype(np.uint8)
        confidence = np.max(prediction_avg, axis=0)

        # Apply confidence threshold
        class_predictions[confidence < self.confidence_threshold] = 0

        # Save outputs
        output_paths = {}

        # Save prediction raster
        pred_path = self.output_dir / f"{output_name}_predictions.tif"
        self._save_prediction_raster(
            class_predictions, pred_path,
            crs, transform, width, height
        )
        output_paths["prediction_raster"] = str(pred_path)

        # Save colored visualization
        vis_path = self.output_dir / f"{output_name}_visualization.tif"
        self._save_visualization(
            class_predictions, vis_path,
            crs, transform, width, height
        )
        output_paths["visualization"] = str(vis_path)

        # Convert to vector (shapefile)
        for class_idx, class_name in enumerate(self.class_names):
            if class_idx == 0:  # Skip background
                continue

            shp_path = self.output_dir / f"{output_name}_{class_name}.shp"
            self._extract_polygons(
                class_predictions, class_idx, class_name,
                shp_path, crs, transform
            )
            output_paths[f"shapefile_{class_name}"] = str(shp_path)

        # Save metadata
        meta_path = self.output_dir / f"{output_name}_metadata.json"
        self._save_metadata(
            input_path, output_paths, class_predictions,
            meta_path, crs
        )
        output_paths["metadata"] = str(meta_path)

        logger.info(f"Feature extraction complete. Outputs saved to: {self.output_dir}")

        return output_paths

    def _generate_windows(
        self,
        width: int,
        height: int
    ) -> List[Window]:
        """Generate sliding windows for inference."""
        windows = []

        for y in range(0, height - self.tile_size + 1, self.stride):
            for x in range(0, width - self.tile_size + 1, self.stride):
                windows.append(Window(x, y, self.tile_size, self.tile_size))

        # Handle edges
        # Right edge
        if width % self.stride != 0:
            for y in range(0, height - self.tile_size + 1, self.stride):
                windows.append(Window(width - self.tile_size, y, self.tile_size, self.tile_size))

        # Bottom edge
        if height % self.stride != 0:
            for x in range(0, width - self.tile_size + 1, self.stride):
                windows.append(Window(x, height - self.tile_size, self.tile_size, self.tile_size))

        # Bottom-right corner
        if width % self.stride != 0 and height % self.stride != 0:
            windows.append(Window(
                width - self.tile_size,
                height - self.tile_size,
                self.tile_size,
                self.tile_size
            ))

        return windows

    def _process_batch(
        self,
        tiles: List[np.ndarray],
        windows: List[Window],
        prediction_sum: np.ndarray,
        prediction_count: np.ndarray
    ):
        """Process a batch of tiles and update prediction accumulator."""
        # Preprocess tiles
        processed_tiles = []
        for tile in tiles:
            transformed = self.transform(image=tile)
            processed_tiles.append(transformed["image"])

        # Stack into batch
        batch = torch.stack(processed_tiles).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(batch)
            probs = F.softmax(outputs, dim=1).cpu().numpy()

        # Update predictions
        for prob, window in zip(probs, windows):
            row_start = window.row_off
            row_end = row_start + window.height
            col_start = window.col_off
            col_end = col_start + window.width

            prediction_sum[:, row_start:row_end, col_start:col_end] += prob
            prediction_count[row_start:row_end, col_start:col_end] += 1

    def _save_prediction_raster(
        self,
        predictions: np.ndarray,
        output_path: Path,
        crs,
        transform,
        width: int,
        height: int
    ):
        """Save predictions as GeoTIFF."""
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(predictions, 1)

        logger.info(f"Prediction raster saved: {output_path}")

    def _save_visualization(
        self,
        predictions: np.ndarray,
        output_path: Path,
        crs,
        transform,
        width: int,
        height: int
    ):
        """Save colored visualization as GeoTIFF."""
        # Create RGB image
        vis = np.zeros((height, width, 3), dtype=np.uint8)

        for class_idx, color in self.class_colors.items():
            mask = predictions == class_idx
            vis[mask] = color

        # Transpose to CHW for rasterio
        vis = np.transpose(vis, (2, 0, 1))

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(vis)

        logger.info(f"Visualization saved: {output_path}")

    def _extract_polygons(
        self,
        predictions: np.ndarray,
        class_idx: int,
        class_name: str,
        output_path: Path,
        crs,
        transform
    ):
        """Extract polygons for a class and save as shapefile."""
        # Create binary mask for class
        mask = (predictions == class_idx).astype(np.uint8)

        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Extract polygons
        polygons = []
        properties = []

        for geom, value in shapes(mask, transform=transform):
            if value == 1:
                polygon = shape(geom)

                # Filter small polygons
                if class_name.startswith("building"):
                    min_area = self.config["inference"]["min_building_area"]
                elif class_name == "road":
                    min_area = self.config["inference"]["min_road_length"]
                else:
                    min_area = 1

                if polygon.area >= min_area:
                    polygons.append(polygon)
                    properties.append({
                        "class": class_name,
                        "area_sqm": polygon.area,
                        "perimeter_m": polygon.length
                    })

        if polygons:
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                properties,
                geometry=polygons,
                crs=crs
            )

            # Save shapefile
            gdf.to_file(output_path)
            logger.info(f"Shapefile saved: {output_path} ({len(polygons)} polygons)")
        else:
            logger.info(f"No polygons found for class: {class_name}")

    def _save_metadata(
        self,
        input_path: Path,
        output_paths: Dict[str, str],
        predictions: np.ndarray,
        output_path: Path,
        crs
    ):
        """Save extraction metadata."""
        # Calculate statistics
        unique, counts = np.unique(predictions, return_counts=True)
        class_stats = {}
        total_pixels = predictions.size

        for idx, count in zip(unique, counts):
            class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            class_stats[class_name] = {
                "pixel_count": int(count),
                "percentage": float(count / total_pixels * 100)
            }

        metadata = {
            "input_file": str(input_path),
            "output_files": output_paths,
            "crs": str(crs),
            "tile_size": self.tile_size,
            "stride": self.stride,
            "confidence_threshold": self.confidence_threshold,
            "class_statistics": class_stats
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved: {output_path}")


class BatchInference:
    """
    Batch inference on multiple GeoTIFF files.
    """

    def __init__(
        self,
        config: Dict,
        model_path: str,
        device: str = "cpu"
    ):
        self.extractor = FeatureExtractor(config, model_path, device)

    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Process all GeoTIFF files in a directory.

        Args:
            input_dir: Directory containing GeoTIFF files
            output_dir: Output directory (optional)

        Returns:
            List of output path dictionaries
        """
        input_dir = Path(input_dir)

        # Find all TIFF files
        tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
        logger.info(f"Found {len(tiff_files)} TIFF files")

        all_outputs = []

        for tiff_file in tqdm(tiff_files, desc="Processing files"):
            try:
                outputs = self.extractor.extract_features(
                    str(tiff_file),
                    tiff_file.stem
                )
                all_outputs.append(outputs)

                # Memory cleanup
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing {tiff_file}: {e}")
                continue

        return all_outputs


def run_inference(
    config_path: str,
    model_path: str,
    input_path: str,
    output_dir: Optional[str] = None
):
    """
    Run inference on drone images.

    Args:
        config_path: Path to configuration file
        model_path: Path to trained model
        input_path: Path to input GeoTIFF or directory
        output_dir: Output directory (optional)
    """
    import yaml

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if output_dir:
        config["inference"]["output_dir"] = output_dir

    # Determine device
    device = config["hardware"]["device"]

    input_path = Path(input_path)

    if input_path.is_file():
        # Single file
        extractor = FeatureExtractor(config, model_path, device)
        extractor.extract_features(str(input_path))
    else:
        # Directory
        batch_inference = BatchInference(config, model_path, device)
        batch_inference.process_directory(str(input_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run feature extraction inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input GeoTIFF or directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    run_inference(args.config, args.model, args.input, args.output)

