"""
Generate Masks from Existing Tiles
===================================
Creates segmentation masks from your actual tile dataset.
Uses image analysis to auto-segment tiles into classes:
- 0: Background
- 1: Building (high intensity)
- 5: Road (medium gray intensity)
- 6: Water (blue-ish tone)
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_tile_and_create_mask(tile: np.ndarray) -> np.ndarray:
    """
    Analyze a tile and create a segmentation mask.
    
    Args:
        tile: Input tile image (H, W, 3) with RGB channels
        
    Returns:
        Segmentation mask (H, W) with class labels
    """
    h, w = tile.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    
    # Extract channels
    r, g, b = tile[:,:,0], tile[:,:,1], tile[:,:,2]
    
    # =====================================================
    # Water Detection (Blue-ish tone, low saturation)
    # =====================================================
    blue_mask = (b > r + 20) & (b > g + 20)  # Blue channel dominant
    water_mask = blue_mask & (hsv[:,:,1] < 100)  # Low saturation
    mask[water_mask] = 6
    
    # =====================================================
    # Road Detection (Gray tone - R≈G≈B)
    # =====================================================
    gray_tolerance = 30
    is_gray = (np.abs(r.astype(int) - g.astype(int)) < gray_tolerance) & \
              (np.abs(g.astype(int) - b.astype(int)) < gray_tolerance)
    
    # Roads are medium gray (not too dark, not too light)
    is_medium_gray = is_gray & (gray > 80) & (gray < 180)
    road_mask = is_medium_gray & ~water_mask
    mask[road_mask] = 5
    
    # =====================================================
    # Building Detection (High intensity, reddish/brownish)
    # =====================================================
    # Buildings have high red or brown tones
    is_bright = gray > 150
    is_reddish = (r > g) | (r > b)
    is_brownish = (r > 100) & (g > 80) & (b < 100)
    
    building_mask = (is_bright & is_reddish) | is_brownish
    # Exclude water and roads
    building_mask = building_mask & ~water_mask & ~road_mask
    mask[building_mask] = 1
    
    # =====================================================
    # Background (everything else)
    # =====================================================
    background_mask = (mask == 0)
    mask[background_mask] = 0
    
    return mask


def generate_masks_for_village(tiles_dir: Path, masks_dir: Path) -> int:
    """
    Generate masks for all tiles in a village directory.
    
    Args:
        tiles_dir: Directory containing .npy tile files
        masks_dir: Output directory for mask files
        
    Returns:
        Number of masks created
    """
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    tile_files = sorted(tiles_dir.glob("tile_*.npy"))
    logger.info(f"Found {len(tile_files)} tiles in {tiles_dir}")
    
    count = 0
    for tile_file in tqdm(tile_files, desc="Generating masks"):
        try:
            # Load tile
            tile = np.load(tile_file)
            
            # Ensure it's RGB (H, W, 3)
            if tile.ndim != 3 or tile.shape[2] != 3:
                logger.warning(f"Skipping {tile_file}: wrong shape {tile.shape}")
                continue
            
            # Generate mask
            mask = analyze_tile_and_create_mask(tile)
            
            # Save mask with same name
            mask_file = masks_dir / tile_file.name
            np.save(mask_file, mask)
            count += 1
            
        except Exception as e:
            logger.error(f"Error processing {tile_file}: {e}")
            continue
    
    return count


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate segmentation masks from tile dataset"
    )
    parser.add_argument(
        "--tiles-dir",
        type=str,
        default="data/tiles",
        help="Root tiles directory containing village folders"
    )
    parser.add_argument(
        "--village-name",
        type=str,
        default=None,
        help="Specific village name (optional, processes all if not specified)"
    )
    
    args = parser.parse_args()
    
    tiles_root = Path(args.tiles_dir)
    
    if not tiles_root.exists():
        logger.error(f"Tiles directory not found: {tiles_root}")
        return
    
    # If village name specified, process only that one
    if args.village_name:
        village_tiles_dir = tiles_root / args.village_name / "tiles"
        village_masks_dir = tiles_root / args.village_name / "masks"
        
        if not village_tiles_dir.exists():
            logger.error(f"Village tiles dir not found: {village_tiles_dir}")
            return
        
        logger.info(f"Processing village: {args.village_name}")
        count = generate_masks_for_village(village_tiles_dir, village_masks_dir)
        logger.info(f"✓ Created {count} masks in {village_masks_dir}")
    
    else:
        # Process all village directories
        total_count = 0
        
        for village_dir in tiles_root.iterdir():
            if not village_dir.is_dir():
                continue
            
            village_tiles_dir = village_dir / "tiles"
            village_masks_dir = village_dir / "masks"
            
            if not village_tiles_dir.exists():
                continue
            
            village_name = village_dir.name
            logger.info(f"\nProcessing village: {village_name}")
            count = generate_masks_for_village(village_tiles_dir, village_masks_dir)
            logger.info(f"✓ Created {count} masks")
            total_count += count
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Total masks created: {total_count}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
