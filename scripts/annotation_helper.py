"""
Annotation Helper Script
========================
Tools for creating and managing ground truth annotations
using automated pre-labeling and manual refinement.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_initial_masks(
    tiles_dir: str,
    output_dir: str,
    threshold_method: str = "otsu"
):
    """
    Create initial segmentation masks using classical CV methods.
    These serve as starting points for manual annotation refinement.

    Args:
        tiles_dir: Directory containing tile images
        output_dir: Output directory for masks
        threshold_method: Thresholding method
    """
    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all tile files
    tile_files = list(tiles_dir.glob("*.npy")) + list(tiles_dir.glob("*.png"))

    logger.info(f"Processing {len(tile_files)} tiles...")

    for tile_path in tqdm(tile_files, desc="Creating initial masks"):
        # Load tile
        if tile_path.suffix == ".npy":
            tile = np.load(tile_path)
        else:
            tile = cv2.imread(str(tile_path))
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        # Create mask using edge detection and thresholding
        mask = _create_initial_mask(tile, threshold_method)

        # Save mask
        mask_path = output_dir / f"{tile_path.stem}.png"
        cv2.imwrite(str(mask_path), mask)

    logger.info(f"Initial masks saved to: {output_dir}")


def _create_initial_mask(
    image: np.ndarray,
    method: str = "otsu"
) -> np.ndarray:
    """
    Create initial mask using classical methods.

    Args:
        image: Input RGB image
        method: Thresholding method

    Returns:
        Initial mask (building/non-building)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Morphological operations to close edges
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Thresholding
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    # Combine edges and thresholding
    combined = cv2.bitwise_or(closed, binary)

    # Clean up
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Find contours and create mask
    mask = np.zeros_like(gray)
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours by area and draw
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area threshold
            # Mark as potential building (class 1)
            cv2.drawContours(mask, [contour], -1, 1, -1)

    return mask


def detect_water_bodies(
    tiles_dir: str,
    output_dir: str
):
    """
    Detect water bodies using color-based segmentation.

    Args:
        tiles_dir: Directory containing tile images
        output_dir: Output directory for water masks
    """
    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_files = list(tiles_dir.glob("*.npy")) + list(tiles_dir.glob("*.png"))

    for tile_path in tqdm(tile_files, desc="Detecting water bodies"):
        if tile_path.suffix == ".npy":
            tile = np.load(tile_path)
        else:
            tile = cv2.imread(str(tile_path))
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        # Convert to HSV
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)

        # Define water color range (blue-ish)
        lower_water = np.array([90, 50, 50])
        upper_water = np.array([130, 255, 255])

        # Create water mask
        water_mask = cv2.inRange(hsv, lower_water, upper_water)

        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

        # Mark as water body class (6)
        water_mask[water_mask > 0] = 6

        # Save
        mask_path = output_dir / f"{tile_path.stem}_water.png"
        cv2.imwrite(str(mask_path), water_mask)

    logger.info(f"Water masks saved to: {output_dir}")


def detect_roads(
    tiles_dir: str,
    output_dir: str
):
    """
    Detect roads using line detection and color analysis.

    Args:
        tiles_dir: Directory containing tile images
        output_dir: Output directory for road masks
    """
    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_files = list(tiles_dir.glob("*.npy")) + list(tiles_dir.glob("*.png"))

    for tile_path in tqdm(tile_files, desc="Detecting roads"):
        if tile_path.suffix == ".npy":
            tile = np.load(tile_path)
        else:
            tile = cv2.imread(str(tile_path))
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Hough lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 50,
            minLineLength=30, maxLineGap=10
        )

        # Create road mask
        road_mask = np.zeros_like(gray)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(road_mask, (x1, y1), (x2, y2), 5, thickness=10)

        # Dilate to create road regions
        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv2.dilate(road_mask, kernel, iterations=2)

        # Save
        mask_path = output_dir / f"{tile_path.stem}_road.png"
        cv2.imwrite(str(mask_path), road_mask)

    logger.info(f"Road masks saved to: {output_dir}")


def merge_masks(
    masks_dir: str,
    output_dir: str
):
    """
    Merge individual class masks into combined annotation masks.

    Args:
        masks_dir: Directory containing individual masks
        output_dir: Output directory for merged masks
    """
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find unique tile names
    all_files = list(masks_dir.glob("*.png"))
    tile_names = set()

    for f in all_files:
        name = f.stem.replace("_water", "").replace("_road", "").replace("_building", "")
        tile_names.add(name)

    logger.info(f"Merging masks for {len(tile_names)} tiles...")

    for tile_name in tqdm(tile_names, desc="Merging masks"):
        # Initialize combined mask
        combined = None

        # Load and merge individual masks
        for class_type, class_value in [("building", 1), ("road", 5), ("water", 6)]:
            mask_path = masks_dir / f"{tile_name}_{class_type}.png"

            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                if combined is None:
                    combined = np.zeros_like(mask)

                # Add class to combined mask (later classes overwrite)
                combined[mask > 0] = class_value

        if combined is not None:
            output_path = output_dir / f"{tile_name}.png"
            cv2.imwrite(str(output_path), combined)

    logger.info(f"Merged masks saved to: {output_dir}")


def create_annotation_project(
    tiles_dir: str,
    project_dir: str,
    project_name: str = "drone_annotation"
):
    """
    Create annotation project structure for use with Label Studio.

    Args:
        tiles_dir: Directory containing tile images
        project_dir: Output project directory
        project_name: Name of the project
    """
    tiles_dir = Path(tiles_dir)
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Find all tiles
    tile_files = list(tiles_dir.glob("*.npy")) + list(tiles_dir.glob("*.png"))

    # Convert .npy to .png for annotation
    images_dir = project_dir / "images"
    images_dir.mkdir(exist_ok=True)

    tasks = []

    for tile_path in tqdm(tile_files, desc="Creating annotation project"):
        if tile_path.suffix == ".npy":
            tile = np.load(tile_path)
            output_path = images_dir / f"{tile_path.stem}.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        else:
            import shutil
            output_path = images_dir / tile_path.name
            shutil.copy(tile_path, output_path)

        # Create task for Label Studio
        tasks.append({
            "data": {
                "image": f"/data/local-files/?d=images/{output_path.name}"
            }
        })

    # Save tasks
    tasks_path = project_dir / "tasks.json"
    with open(tasks_path, 'w') as f:
        json.dump(tasks, f, indent=2)

    # Create Label Studio config
    label_config = """
<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="brush" toName="image">
    <Label value="Building_RCC" background="rgba(255, 0, 0, 0.7)"/>
    <Label value="Building_Tiled" background="rgba(0, 255, 0, 0.7)"/>
    <Label value="Building_Tin" background="rgba(0, 0, 255, 0.7)"/>
    <Label value="Building_Others" background="rgba(255, 255, 0, 0.7)"/>
    <Label value="Road" background="rgba(128, 128, 128, 0.7)"/>
    <Label value="Waterbody" background="rgba(0, 255, 255, 0.7)"/>
  </BrushLabels>
</View>
    """

    config_path = project_dir / "label_config.xml"
    with open(config_path, 'w') as f:
        f.write(label_config)

    logger.info(f"Annotation project created at: {project_dir}")
    logger.info(f"Tasks: {len(tasks)}")
    logger.info("To use with Label Studio:")
    logger.info(f"  1. Start Label Studio: label-studio start")
    logger.info(f"  2. Import tasks from: {tasks_path}")
    logger.info(f"  3. Use label config from: {config_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotation helper tools")
    parser.add_argument("command", choices=["initial", "water", "roads", "merge", "project"])
    parser.add_argument("--tiles-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    if args.command == "initial":
        create_initial_masks(args.tiles_dir, args.output_dir)
    elif args.command == "water":
        detect_water_bodies(args.tiles_dir, args.output_dir)
    elif args.command == "roads":
        detect_roads(args.tiles_dir, args.output_dir)
    elif args.command == "merge":
        merge_masks(args.tiles_dir, args.output_dir)
    elif args.command == "project":
        create_annotation_project(args.tiles_dir, args.output_dir)
