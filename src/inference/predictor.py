"""
Inference Pipeline for Feature Extraction
==========================================
Sliding window inference on large GeoTIFF files
with prediction stitching and post-processing.
Supports resumable inference with checkpointing.
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
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
import cv2
from src.models.segmentation import load_model as load_pytorch_model
from src.inference.onnx_inference import load_onnx_model

from ..models import load_model, create_model
from ..preprocessing import get_validation_augmentation
from .gis_export import GISExporter

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
        device: str = "cpu",
        model_type="pytorch"
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
        self.model_type = model_type

        # Load primary model
        if model_type == "onnx":
            self.model = load_onnx_model(model_path)
        else:
            self.model = load_pytorch_model(model_path, config, device)
            self.model.eval()

        # Load class-specific models (waterbody specialized model)
        self.waterbody_model = None
        self._load_class_specific_models(config, device, model_type)

        # Inference settings
        self.tile_size = config["data"]["tile_size"]
        self.stride = config["inference"]["stride"]
        self.batch_size = config["inference"]["batch_size"]
        self.confidence_threshold = config["inference"]["confidence_threshold"]

        # Class information
        self.num_classes = config["data"]["num_seg_classes"]
        self.class_names = list(config["data"]["segmentation_classes"].values())
        self.class_colors = self._get_class_colors()
        
        # Create mapping from class name to index
        self.class_name_to_idx = {
            name: idx for idx, name in enumerate(self.class_names)
        }
        self.waterbody_idx = self.class_name_to_idx.get("waterbody", 6)

        # Preprocessing
        self.transform = get_validation_augmentation(config)

        # Output settings
        self.output_dir = Path(config["inference"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GIS exporter
        self.gis_exporter = GISExporter(
            output_dir=str(self.output_dir),
            crs=None,  # Will be set from source data
            min_polygon_area=config["inference"].get("min_building_area", 10.0),
            min_line_length=config["inference"].get("min_road_length", 5.0),
            config=config
        )

        logger.info("Feature extractor initialized")
        logger.info(f"Tile size: {self.tile_size}, Stride: {self.stride}")
        if self.waterbody_model:
            logger.info("Waterbody class-specific model loaded")

    def _get_village_output_dir(self, output_name: str) -> Path:
        """Return the per-village output directory."""
        village_output_dir = self.output_dir / output_name
        village_output_dir.mkdir(parents=True, exist_ok=True)
        return village_output_dir

    def _create_gis_exporter(self, output_dir: Path, crs=None) -> GISExporter:
        """Create a GIS exporter scoped to a single village output directory."""
        return GISExporter(
            output_dir=str(output_dir),
            crs=crs,
            min_polygon_area=self.config["inference"].get("min_building_area", 10.0),
            min_line_length=self.config["inference"].get("min_road_length", 5.0),
            config=self.config
        )

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

    def _save_inference_checkpoint(
            self,
            checkpoint_path: Path,
            processed_window_indices: set,
            input_path: Path,
            output_name: str
    ):
        """Save inference progress checkpoint for resumable processing."""
        from datetime import datetime

        checkpoint_data = {
            "input_file": str(input_path),
            "output_name": output_name,
            "processed_window_indices": list(processed_window_indices),
            "timestamp": datetime.now().isoformat()
        }

        # Ensure parent directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.debug(
            f"Saved inference checkpoint with "
            f"{len(processed_window_indices)} processed windows"
        )

    def extract_features(
        self,
        input_path: str,
        output_name: Optional[str] = None,
        resume_from_checkpoint: bool = True
    ) -> Dict[str, str]:
        """
        Extract features from a large GeoTIFF file.
        Supports resuming from checkpoints if processing was interrupted.

        Args:
            input_path: Path to input GeoTIFF
            output_name: Name for output files
            resume_from_checkpoint: If True, resume from last saved checkpoint

        Returns:
            Dictionary with output file paths
        """
        input_path = Path(input_path)

        if output_name is None:
            output_name = input_path.stem

        logger.info(f"Processing: {input_path}")
        village_output_dir = self._get_village_output_dir(output_name)

        # Check for existing inference checkpoint
        checkpoint_path = village_output_dir / f"{output_name}_inference_checkpoint.json"
        inference_state = None
        
        if resume_from_checkpoint and checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            with open(checkpoint_path) as f:
                inference_state = json.load(f)

        with rasterio.open(input_path) as src:
            # Get image properties
            width = src.width
            height = src.height
            crs = src.crs
            transform = src.transform

            # Update GIS exporter with source CRS
            if crs is not None:
                gis_exporter = self._create_gis_exporter(village_output_dir, crs)
            else:
                logger.warning("No CRS found in source, using WGS84")
                gis_exporter = self._create_gis_exporter(
                    village_output_dir,
                    CRS.from_epsg(4326)
                )

            logger.info(f"Image size: {width} x {height}")
            logger.info(f"CRS: {crs}")

            # Initialize prediction accumulator
            # OPTIMIZATION: Check for existing partial results
            if inference_state and "prediction_accumulator" in inference_state:
                logger.info("Loading partial predictions from checkpoint...")
                class_map = np.zeros((height, width), dtype=np.uint8)
                confidence_map = np.zeros((height, width), dtype=np.float16)
                processed_window_indices = set(inference_state.get("processed_window_indices", []))
            else:
                class_map = np.zeros((height, width), dtype=np.uint8)
                confidence_map = np.zeros((height, width), dtype=np.float16)
                processed_window_indices = set()

            # Generate windows
            windows = self._generate_windows(width, height)
            logger.info(f"Total windows: {len(windows)}")
            
            if processed_window_indices:
                logger.info(f"Resuming from window {len(processed_window_indices)}/{len(windows)}")

            # Process in batches
            batch_tiles = []
            batch_windows = []
            batch_indices = []

            for window_idx, window in enumerate(tqdm(windows, desc="Extracting features")):
                
                # Skip already processed windows
                if window_idx in processed_window_indices:
                    continue
                
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
                batch_indices.append(window_idx)

                # Process batch
                if len(batch_tiles) >= self.batch_size:
                    self._process_batch(
                        batch_tiles, batch_windows,
                        class_map, confidence_map
                    )
                    processed_window_indices.update(batch_indices)
                    
                    # OPTIMIZATION: Save checkpoint every 50 batches
                    if len(processed_window_indices) % (self.batch_size * 50) == 0:
                        self._save_inference_checkpoint(
                            checkpoint_path, processed_window_indices,
                            input_path, output_name
                        )
                    
                    batch_tiles = []
                    batch_windows = []
                    batch_indices = []

                    # Memory cleanup
                    gc.collect()

            # Process remaining tiles
            if batch_tiles:
                self._process_batch(
                    batch_tiles, batch_windows,
                    class_map, confidence_map
                )
                processed_window_indices.update(batch_indices)

        # Class predictions
        class_predictions = class_map
        confidence = confidence_map.astype(np.float32)

        # Apply confidence threshold
        class_predictions[confidence < self.confidence_threshold] = 0

        # Save outputs
        output_paths = {}

        # Save prediction raster
        pred_path = village_output_dir / f"{output_name}_predictions.tif"
        self._save_prediction_raster(
            class_predictions, pred_path,
            crs, transform, width, height
        )
        output_paths["prediction_raster"] = str(pred_path)

        # Save colored visualization
        vis_path = village_output_dir / f"{output_name}_visualization.tif"
        self._save_visualization(
            class_predictions, vis_path,
            crs, transform, width, height
        )
        output_paths["visualization"] = str(vis_path)

        # Export to GIS formats (Shapefile + GeoPackage)
        logger.info("Exporting predictions to GIS formats...")
        gis_outputs = gis_exporter.export_predictions(
            class_predictions,
            transform,
            output_name,
            confidence
        )
        output_paths.update(gis_outputs)

        # Save metadata
        meta_path = village_output_dir / f"{output_name}_metadata.json"
        self._save_metadata(
            input_path, output_paths, class_predictions,
            meta_path, crs
        )
        output_paths["metadata"] = str(meta_path)

        logger.info(f"Feature extraction complete. Outputs saved to: {village_output_dir}")

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
            class_map: np.ndarray,
            confidence_map: np.ndarray
    ):
        """
        Process a batch of tiles with class-specific model inference.
        - Primary model for all classes
        - Waterbody model replaces waterbody channel if available
        - Empty/no-data regions masked to prevent false waterbody detection
        """

        processed_tiles = []
        no_data_masks = []  # Track empty regions for each tile
        
        for tile in tiles:
            # Detect no-data/empty regions (all zeros or near-zero)
            # Valid tiles should have at least some non-zero pixel values
            tile_gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            no_data_mask = tile_gray < 5  # Threshold for empty pixels
            no_data_masks.append(no_data_mask)
            
            transformed = self.transform(image=tile)
            processed_tiles.append(transformed["image"])

        # Stack tiles into a batch tensor
        batch = torch.stack(processed_tiles)

        # ==============================
        # 🔹 Primary Model Inference
        # ==============================
        if self.model_type == "onnx":
            # Convert to NumPy for ONNX Runtime
            input_batch = batch.cpu().numpy()
            input_name = self.model.get_inputs()[0].name

            outputs = self.model.run(None, {input_name: input_batch})
            logits = outputs[0]

            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        else:
            # PyTorch inference
            batch = batch.to(self.device)
            with torch.no_grad():
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1).cpu().numpy()

        # ==============================
        # 🔹 Class-Specific Model Inference (Waterbody)
        # ==============================
        if self.waterbody_model is not None:
            try:
                if self.model_type == "onnx":
                    input_batch = batch.cpu().numpy()
                    input_name = self.waterbody_model.get_inputs()[0].name
                    
                    waterbody_outputs = self.waterbody_model.run(
                        None,
                        {input_name: input_batch}
                    )
                    waterbody_logits = waterbody_outputs[0]
                    
                    # Apply softmax
                    exp_logits = np.exp(
                        waterbody_logits - np.max(waterbody_logits, axis=1, keepdims=True)
                    )
                    waterbody_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                else:
                    # PyTorch inference
                    batch_device = batch.to(self.device)
                    with torch.no_grad():
                        waterbody_outputs = self.waterbody_model(batch_device)
                        waterbody_probs = F.softmax(waterbody_outputs, dim=1).cpu().numpy()
                
                # Replace waterbody channel (index 6) with specialized model predictions
                # BUT: Only where data is valid (not empty/no-data regions)
                for i, no_data_mask in enumerate(no_data_masks):
                    # Set waterbody probability to 0 in no-data regions
                    waterbody_probs[i, self.waterbody_idx, no_data_mask] = 0
                    # Boost background probability in no-data regions
                    waterbody_probs[i, 0, no_data_mask] = 1.0
                
                probs[:, self.waterbody_idx, :, :] = waterbody_probs[:, self.waterbody_idx, :, :]
                
            except Exception as e:
                logger.warning(f"Waterbody model inference failed, using primary model: {e}")
                # Fall back to primary model predictions

        # ==============================
        # 🔹 Mask Empty Regions (No-Data)
        # ==============================
        for i, no_data_mask in enumerate(no_data_masks):
            # Set all non-background probabilities to 0 in empty regions
            probs[i, 1:, no_data_mask] = 0
            # Ensure background (class 0) has highest probability in empty regions
            probs[i, 0, no_data_mask] = 1.0

        # ==============================
        # 🔹 Update Prediction Maps
        # ==============================
        for prob, window, no_data_mask in zip(probs, windows, no_data_masks):
            row_start = window.row_off
            row_end = row_start + window.height
            col_start = window.col_off
            col_end = col_start + window.width

            pred_class = np.argmax(prob, axis=0).astype(np.uint8)
            pred_conf = np.max(prob, axis=0).astype(np.float16)

            # Force empty regions to background class
            pred_class[no_data_mask] = 0
            pred_conf[no_data_mask] = 0.0

            # Update only where confidence is higher
            existing_conf = confidence_map[row_start:row_end, col_start:col_end]
            mask = pred_conf > existing_conf

            class_map[row_start:row_end, col_start:col_end][mask] = pred_class[mask]
            confidence_map[row_start:row_end, col_start:col_end][mask] = pred_conf[mask]

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

    def _save_metadata(
            self,
            input_path: Path,
            output_paths: Dict[str, str],
            predictions: np.ndarray,
            output_path: Path,
            crs
    ):
        """Save extraction metadata."""
        # Convert Path objects to strings
        input_path = str(input_path)
        output_path = str(output_path)

        # Ensure all output paths are JSON serializable
        output_paths_serializable = {
            key: str(value) for key, value in output_paths.items()
        }

        # Calculate statistics
        unique, counts = np.unique(predictions, return_counts=True)
        class_stats = {}
        total_pixels = int(predictions.size)

        for idx, count in zip(unique, counts):
            idx = int(idx)
            count = int(count)
            class_name = (
                self.class_names[idx]
                if idx < len(self.class_names)
                else f"class_{idx}"
            )

            class_stats[class_name] = {
                "pixel_count": count,
                "percentage": float(count / total_pixels * 100)
            }

        metadata = {
            "input_file": input_path,
            "output_files": output_paths_serializable,
            "crs": str(crs),
            "tile_size": int(self.tile_size),
            "stride": int(self.stride),
            "confidence_threshold": float(self.confidence_threshold),
            "class_statistics": class_stats,
        }

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved: {output_path}")

    def _load_class_specific_models(
        self,
        config: Dict,
        device: str,
        model_type: str
    ):
        """
        Load class-specific specialized models.
        Currently loads waterbody model from batch_3_best.pth or batch_3_best.onnx if available.
        """
        # Try to find waterbody-specific model (.onnx or .pth)
        checkpoint_dir = Path("outputs/checkpoints")
        waterbody_pth = checkpoint_dir / "batch_3_best.pth"
        waterbody_onnx = checkpoint_dir / "batch_3_best.onnx"
        
        waterbody_model_path = None
        waterbody_model_format = None
        
        # Prefer .onnx if available, otherwise use .pth
        if waterbody_onnx.exists():
            waterbody_model_path = waterbody_onnx
            waterbody_model_format = "onnx"
        elif waterbody_pth.exists():
            waterbody_model_path = waterbody_pth
            waterbody_model_format = "pytorch"
        
        if waterbody_model_path is not None:
            try:
                if waterbody_model_format == "onnx":
                    self.waterbody_model = load_onnx_model(str(waterbody_model_path))
                else:
                    self.waterbody_model = load_pytorch_model(
                        str(waterbody_model_path),
                        config,
                        device
                    )
                    self.waterbody_model.eval()
                
                logger.info(f"Loaded waterbody-specific model: {waterbody_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load waterbody model: {e}")
                self.waterbody_model = None
        else:
            logger.debug(f"Waterbody model not found at {checkpoint_dir / 'batch_3_best.*'}")
            self.waterbody_model = None


class BatchInference:
    """
    Batch inference on multiple GeoTIFF files.
    """

    def __init__(
        self,
        config: Dict,
        model_path: str,
        device: str = "cpu",
        model_type: str = "pytorch"
    ):
        self.extractor = FeatureExtractor(config, model_path, device, model_type)

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

    # Determine model type
    model_ext = Path(model_path).suffix.lower()
    if model_ext == ".onnx":
        model_type = "onnx"
    elif model_ext in [".pth", ".pt"]:
        model_type = "pytorch"
    else:
        raise ValueError(f"Unsupported model format: {model_ext}")

    logger.info(f"Using {model_type.upper()} model for inference: {model_path}")

    input_path = Path(input_path)

    if input_path.is_file():
        # Single file
        extractor = FeatureExtractor(config, model_path, device, model_type)
        extractor.extract_features(str(input_path))
    else:
        # Directory
        batch_inference = BatchInference(config, model_path, device, model_type)
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

