"""
Script to Load and View All .npy Files
======================================
This script loads all tile and mask .npy files and displays them
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import colorize_prediction, create_overlay

def load_all_data(data_dir: str = "scripts/data") -> tuple:
    """Load all tiles and masks from the generated dataset."""
    data_dir = Path(data_dir)
    
    tiles_data = {}
    masks_data = {}
    
    tiles_base = data_dir / "tiles"
    annotations_base = data_dir / "annotations"
    
    print("=" * 70)
    print("LOADING ALL .NPY FILES")
    print("=" * 70)
    
    if not tiles_base.exists():
        print(f"ERROR: Tiles directory not found: {tiles_base}")
        return None, None
    
    # Find all villages
    villages = sorted([d.name for d in tiles_base.iterdir() if d.is_dir()])
    print(f"\nFound villages: {villages}\n")
    
    total_tiles = 0
    total_masks = 0
    
    for village in villages:
        tiles_village_dir = tiles_base / village / "tiles"
        masks_village_dir = annotations_base / village / "masks"
        
        tiles_data[village] = {}
        masks_data[village] = {}
        
        # Load tiles
        if tiles_village_dir.exists():
            tile_files = sorted(list(tiles_village_dir.glob("*.npy")))
            print(f"Loading {village} tiles...")
            
            for tile_file in tile_files:
                try:
                    data = np.load(tile_file)
                    tiles_data[village][tile_file.stem] = data
                    total_tiles += 1
                    print(f"  ✓ {tile_file.name} - shape: {data.shape}, dtype: {data.dtype}")
                except Exception as e:
                    print(f"  ✗ Failed to load {tile_file.name}: {e}")
        
        # Load masks
        if masks_village_dir.exists():
            mask_files = sorted(list(masks_village_dir.glob("*.npy")))
            print(f"Loading {village} masks...")
            
            for mask_file in mask_files:
                try:
                    data = np.load(mask_file)
                    masks_data[village][mask_file.stem] = data
                    total_masks += 1
                    print(f"  ✓ {mask_file.name} - shape: {data.shape}, dtype: {data.dtype}, classes: {np.unique(data)}")
                except Exception as e:
                    print(f"  ✗ Failed to load {mask_file.name}: {e}")
        
        print()
    
    # Print summary
    print("=" * 70)
    print("LOADING SUMMARY")
    print("=" * 70)
    print(f"Total tiles loaded: {total_tiles}")
    print(f"Total masks loaded: {total_masks}")
    print(f"Total villages: {len(villages)}\n")
    
    return tiles_data, masks_data


def display_sample_tiles(tiles_data: Dict, masks_data: Dict, num_samples: int = 9):
    """Display sample tiles with their masks and overlays."""
    
    if not tiles_data:
        print("No data to display!")
        return
    
    # Get first village
    first_village = list(tiles_data.keys())[0]
    tiles = tiles_data[first_village]
    masks = masks_data[first_village]
    
    # Get first N tiles
    tile_names = sorted(list(tiles.keys()))[:num_samples]
    
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    print(f"\nDisplaying {num_samples} sample tiles from {first_village}...\n")
    
    for idx, tile_name in enumerate(tile_names):
        tile = tiles[tile_name]
        mask = masks.get(tile_name)
        
        if mask is not None:
            # Create colored mask
            colored_mask = colorize_prediction(mask)
            
            # Normalize tile if needed
            if tile.max() > 1:
                tile_normalized = tile / 255.0
            else:
                tile_normalized = tile
            
            axes[idx].imshow(colored_mask)
            axes[idx].set_title(f"{tile_name}\nShape: {mask.shape}, Classes: {len(np.unique(mask))}")
        else:
            axes[idx].imshow(tile)
            axes[idx].set_title(f"{tile_name}\nShape: {tile.shape}")
        
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(tile_names), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Sample Tiles and Masks from {first_village}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def display_single_tile_with_overlay(tiles_data: Dict, masks_data: Dict, village: str = None, tile_idx: int = 0):
    """Display a single tile with mask and overlay comparison."""
    
    if not tiles_data:
        print("No data to display!")
        return
    
    if village is None:
        village = list(tiles_data.keys())[0]
    
    tiles = tiles_data[village]
    masks = masks_data[village]
    
    tile_name = sorted(list(tiles.keys()))[tile_idx]
    tile = tiles[tile_name]
    mask = masks.get(tile_name)
    
    if mask is None:
        print(f"Mask not found for {tile_name}")
        return
    
    # Normalize tile if needed
    if tile.max() > 1:
        tile = (tile / 255.0 * 255).astype(np.uint8)
    else:
        tile = (tile * 255).astype(np.uint8)
    
    if len(tile.shape) == 2:
        tile = np.stack([tile] * 3, axis=-1)
    
    # Create visualizations
    colored_mask = colorize_prediction(mask)
    overlay = create_overlay(tile, mask, alpha=0.6)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(tile)
    axes[0].set_title("Original Tile Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(colored_mask)
    axes[1].set_title(f"Segmentation Mask\nClasses: {np.unique(mask)}", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Image + Mask)", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f"{village} - {tile_name}\nTile shape: {tile.shape}, Mask shape: {mask.shape}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load all data
    tiles_data, masks_data = load_all_data("scripts/data")
    
    if tiles_data is None:
        print("\nFailed to load data. Please check your data directory.")
        exit(1)
    
    # Display sample tiles
    print("\n" + "=" * 70)
    print("DISPLAYING SAMPLE TILES")
    print("=" * 70)
    display_sample_tiles(tiles_data, masks_data, num_samples=9)
    
    # Display single tile with comparisons
    print("\nDisplaying single tile with overlay comparison...")
    display_single_tile_with_overlay(tiles_data, masks_data, tile_idx=0)
    
    # Print data statistics
    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)
    for village, tiles in tiles_data.items():
        if tiles:
            first_tile = list(tiles.values())[0]
            print(f"\n{village}:")
            print(f"  Number of tiles: {len(tiles)}")
            print(f"  Tile shape: {first_tile.shape}")
            print(f"  Tile data type: {first_tile.dtype}")
            print(f"  Value range: [{first_tile.min()}, {first_tile.max()}]")
            
            if village in masks_data:
                masks = masks_data[village]
                if masks:
                    first_mask = list(masks.values())[0]
                    print(f"  Number of masks: {len(masks)}")
                    print(f"  Mask shape: {first_mask.shape}")
                    print(f"  Mask classes: {np.unique(first_mask)}")

