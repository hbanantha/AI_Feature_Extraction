"""
Generate Sample Data for Testing
=================================
Creates synthetic sample data for testing the pipeline
without requiring actual drone imagery.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_tile(
    size: int = 256,
    num_buildings: int = 3,
    has_road: bool = True,
    has_water: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic tile with buildings, roads, and water.

    Args:
        size: Tile size
        num_buildings: Number of buildings to generate
        has_road: Whether to add a road
        has_water: Whether to add water body

    Returns:
        Tuple of (image, mask)
    """
    # Create base image (greenish/brownish background like terrain)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = np.random.randint(80, 120, (size, size))  # R
    image[:, :, 1] = np.random.randint(100, 140, (size, size))  # G
    image[:, :, 2] = np.random.randint(60, 100, (size, size))  # B

    # Add some noise/texture
    noise = np.random.randint(-20, 20, (size, size, 3))
    image = np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    # Create mask
    mask = np.zeros((size, size), dtype=np.uint8)

    # Add road
    if has_road:
        # Random road orientation
        if random.random() > 0.5:
            # Horizontal road
            y = random.randint(size // 4, 3 * size // 4)
            road_width = random.randint(10, 20)

            image[y:y+road_width, :] = [100, 100, 100]  # Gray
            mask[y:y+road_width, :] = 5  # Road class
        else:
            # Vertical road
            x = random.randint(size // 4, 3 * size // 4)
            road_width = random.randint(10, 20)

            image[:, x:x+road_width] = [100, 100, 100]
            mask[:, x:x+road_width] = 5

    # Add water body
    if has_water:
        cx = random.randint(size // 4, 3 * size // 4)
        cy = random.randint(size // 4, 3 * size // 4)
        rx = random.randint(20, 50)
        ry = random.randint(20, 50)

        cv2.ellipse(image, (cx, cy), (rx, ry), 0, 0, 360, (0, 100, 150), -1)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 6, -1)

    # Add buildings
    for _ in range(num_buildings):
        # Random position and size
        x = random.randint(10, size - 60)
        y = random.randint(10, size - 60)
        w = random.randint(20, 50)
        h = random.randint(20, 50)

        # Random roof type
        roof_type = random.randint(1, 4)

        if roof_type == 1:  # RCC - darker gray/concrete
            color = (150, 150, 150)
        elif roof_type == 2:  # Tiled - terracotta/orange
            color = (60, 80, 180)
        elif roof_type == 3:  # Tin - bright/metallic
            color = (180, 180, 200)
        else:  # Others - variable
            color = (140, 140, 100)

        # Draw building
        cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)
        cv2.rectangle(mask, (x, y), (x+w, y+h), roof_type, -1)

        # Add slight border
        cv2.rectangle(image, (x, y), (x+w, y+h), (50, 50, 50), 1)

    return image, mask


def generate_sample_dataset(
    output_dir: str,
    num_villages: int = 5,
    tiles_per_village: int = 20
):
    """
    Generate a sample dataset for testing.

    Args:
        output_dir: Output directory
        num_villages: Number of villages to generate
        tiles_per_village: Number of tiles per village
    """
    output_dir = Path(output_dir)

    for village_idx in range(num_villages):
        village_name = f"village_{village_idx + 1:02d}"

        # Create directories
        tiles_dir = output_dir / "tiles" / village_name / "tiles"
        masks_dir = output_dir / "annotations" / village_name / "masks"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {village_name}...")

        for tile_idx in range(tiles_per_village):
            # Generate synthetic tile
            num_buildings = random.randint(1, 5)
            has_road = random.random() > 0.3
            has_water = random.random() > 0.8

            image, mask = generate_synthetic_tile(
                size=256,
                num_buildings=num_buildings,
                has_road=has_road,
                has_water=has_water
            )

            # Save tile
            tile_name = f"tile_{tile_idx:04d}_0000"
            np.save(tiles_dir / f"{tile_name}.npy", image)
            np.save(masks_dir / f"{tile_name}.npy", mask)

    logger.info(f"Sample dataset generated at: {output_dir}")
    logger.info(f"Villages: {num_villages}")
    logger.info(f"Tiles per village: {tiles_per_village}")
    logger.info(f"Total tiles: {num_villages * tiles_per_village}")


def generate_sample_geotiff(
    output_path: str,
    width: int = 2000,
    height: int = 2000
):
    """
    Generate a sample GeoTIFF for testing the tiling pipeline.

    Args:
        output_path: Output file path
        width: Image width
        height: Image height
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        logger.error("rasterio not installed. Run: pip install rasterio")
        return

    # Create synthetic large image
    image = np.zeros((3, height, width), dtype=np.uint8)

    # Background
    image[0] = np.random.randint(80, 120, (height, width))
    image[1] = np.random.randint(100, 140, (height, width))
    image[2] = np.random.randint(60, 100, (height, width))

    # Add some features
    for _ in range(50):
        # Buildings
        x = random.randint(0, width - 100)
        y = random.randint(0, height - 100)
        w = random.randint(30, 80)
        h = random.randint(30, 80)

        color = random.choice([
            [150, 150, 150],
            [60, 80, 180],
            [180, 180, 200]
        ])

        image[0, y:y+h, x:x+w] = color[0]
        image[1, y:y+h, x:x+w] = color[1]
        image[2, y:y+h, x:x+w] = color[2]

    # Add roads
    for _ in range(5):
        if random.random() > 0.5:
            y = random.randint(0, height - 20)
            image[:, y:y+15, :] = 100
        else:
            x = random.randint(0, width - 20)
            image[:, :, x:x+15] = 100

    # Create GeoTIFF
    transform = from_origin(0, height, 1, 1)  # Simple pixel coordinates

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=np.uint8,
        crs='EPSG:32643',  # UTM zone 43N (common for India)
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(image)

    logger.info(f"Sample GeoTIFF created: {output_path}")
    logger.info(f"Size: {width} x {height}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample data")
    parser.add_argument("--type", choices=["dataset", "geotiff"], default="dataset")
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--villages", type=int, default=5)
    parser.add_argument("--tiles", type=int, default=20)

    args = parser.parse_args()

    if args.type == "dataset":
        generate_sample_dataset(
            args.output,
            num_villages=args.villages,
            tiles_per_village=args.tiles
        )
    else:
        generate_sample_geotiff(f"{args.output}/sample.tif")
