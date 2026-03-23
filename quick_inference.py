"""
Quick Inference Script - Generate GeoPackage from pretrained model
"""

import argparse
import yaml
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import geopandas as gpd
from shapely.geometry import box, Polygon
import rasterio
from rasterio.transform import Affine
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model_from_config(config: dict, device='cpu'):
    """Create model using config."""
    from src.models import create_model
    
    logger.info("Creating model...")
    model = create_model(config).to(device)
    model.eval()
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def run_inference_on_dataset(model, config, device='cpu'):
    """
    Run inference on all tiles and create predictions GeoPackage.
    
    Returns:
        Path to generated GeoPackage file
    """
    from src.preprocessing import DroneImageDataset, get_validation_augmentation
    
    logger.info("Loading dataset...")
    transform = get_validation_augmentation(config)
    
    dataset = DroneImageDataset(
        tiles_dir=config['data']['tiles_dir'],
        masks_dir=config['data']['annotations_dir'],
        transform=transform,
        is_training=False,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Prepare output directory
    output_dir = Path(config['training']['checkpoint_dir']).parent / 'gis_exports'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store predictions
    features = []
    geometries = []
    
    logger.info("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            paths = batch['path']
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)  # [B, H, W]
            
            batch_size = predictions.shape[0]
            for i in range(batch_size):
                pred = predictions[i].cpu().numpy()
                path = paths[i]
                
                # Extract coordinates from path if available
                # tile_0001_0272.npy -> row=1, col=272
                tile_name = Path(path).stem
                try:
                    parts = tile_name.split('_')
                    if len(parts) >= 3:
                        row = int(parts[-2])
                        col = int(parts[-1])
                    else:
                        row, col = 0, 0
                except:
                    row, col = 0, 0
                
                # Get unique classes in prediction
                unique_classes = np.unique(pred)
                
                for class_id in unique_classes:
                    if class_id == 0:  # Skip background
                        continue
                    
                    # Create mask for this class
                    class_mask = (pred == class_id).astype(np.uint8)
                    
                    # Find contours
                    contours, _ = cv2.findContours(
                        class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    for contour in contours:
                        if cv2.contourArea(contour) < 5:  # Skip small noise
                            continue
                        
                        # Convert contour to coordinate list
                        contour = contour.squeeze()
                        if contour.ndim == 1:
                            continue
                        
                        # Create polygon (tile-relative coordinates)
                        coords = [(c[0], c[1]) for c in contour]
                        if len(coords) >= 3:
                            try:
                                poly = Polygon(coords)
                                geometries.append(poly)
                                features.append({
                                    'class_id': int(class_id),
                                    'class_name': config['data']['segmentation_classes'].get(
                                        class_id, 'unknown'
                                    ),
                                    'tile_row': row,
                                    'tile_col': col,
                                    'confidence': 0.85,  # Placeholder
                                })
                            except Exception as e:
                                logger.warning(f"Failed to create polygon: {e}")
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * batch_size}/{len(dataset)} samples")
    
    # Create GeoDataFrame
    logger.info(f"Creating GeoDataFrame with {len(geometries)} features...")
    gdf = gpd.GeoDataFrame(features, geometry=geometries, crs='EPSG:4326')
    
    # Save to GeoPackage
    gpkg_path = output_dir / 'predicted_features.gpkg'
    logger.info(f"Saving to {gpkg_path}...")
    gdf.to_file(gpkg_path, driver='GPKG')
    
    logger.info(f"✓ GeoPackage created: {gpkg_path}")
    logger.info(f"  - Total features: {len(gdf)}")
    logger.info(f"  - Classes: {gdf['class_name'].unique()}")
    
    return gpkg_path


def main():
    parser = argparse.ArgumentParser(
        description='Quick inference to generate GeoPackage predictions'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to load (optional)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create model
    model = create_model_from_config(config, device=args.device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded")
    else:
        logger.info("Using pretrained model weights (no fine-tuned checkpoint)")
    
    # Run inference
    gpkg_path = run_inference_on_dataset(model, config, device=args.device)
    
    logger.info("✓ Inference complete!")


if __name__ == '__main__':
    main()
