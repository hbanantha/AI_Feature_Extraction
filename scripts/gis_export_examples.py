"""
GIS Export Pipeline Example
===========================
Demonstrates how to use the new GISExporter for generating Shapefiles 
and GeoPackage files from segmentation predictions.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging
from rasterio.crs import CRS
from affine import Affine

from src.inference.gis_export import GISExporter, FEATURE_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_export():
    """
    Basic example: Export predictions to Shapefile and GeoPackage.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Export (Shapefile + GeoPackage)")
    print("="*70)

    # Create synthetic predictions for demonstration
    height, width = 512, 512
    predictions = np.zeros((height, width), dtype=np.uint8)
    
    # Create synthetic features
    # Buildings (class 1)
    predictions[50:150, 50:150] = 1
    predictions[200:300, 50:150] = 2
    predictions[350:450, 50:150] = 3
    
    # Roads (class 5)
    predictions[200:210, 100:400] = 5
    predictions[150:350, 100:110] = 5
    
    # Water (class 6)
    predictions[400:480, 200:350] = 6

    # Create confidence scores
    confidence = np.random.rand(height, width) * 0.5 + 0.5
    
    # Define geotransform (EPSG:32643 - UTM Zone 43N)
    transform = Affine(10.0, 0.0, 500000.0,
                       0.0, -10.0, 2500000.0)
    
    crs = CRS.from_epsg(32643)

    # Initialize exporter
    output_dir = "outputs/gis_exports"
    exporter = GISExporter(
        output_dir=output_dir,
        crs=crs,
        min_polygon_area=50.0,  # 50 m²
        min_line_length=5.0     # 5 m
    )

    # Export predictions
    output_paths = exporter.export_predictions(
        predictions=predictions,
        transform=transform,
        output_name="example_features",
        confidence=confidence
    )

    # Print results
    print("\n✓ Export successful!")
    print("\nOutput files:")
    for file_type, path in output_paths.items():
        if path is not None:
            print(f"  {file_type}: {path}")

    # Validate exports
    print("\nValidating exports...")
    validation = exporter.validate_exports()
    for filename, valid in validation.items():
        status = "✓" if valid else "✗"
        print(f"  {status} {filename}")

    # Print layer summary
    print("\nLayer Summary:")
    summary = exporter.get_layer_summary()
    for layer_name, info in summary.items():
        print(f"  {layer_name}:")
        print(f"    - Features: {info['feature_count']}")
        print(f"    - CRS: {info['crs']}")
        print(f"    - Bounds: {info['bounds']}")

    return output_paths


def example_batch_export():
    """
    Batch example: Export multiple prediction sets and merge into single GeoPackage.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Export with Merge")
    print("="*70)

    output_dir = "outputs/gis_exports_batch"
    exporter = GISExporter(
        output_dir=output_dir,
        crs=CRS.from_epsg(4326),  # WGS84
        min_polygon_area=100.0,
        min_line_length=10.0
    )

    transform = Affine(0.0001, 0.0, 0.0,
                       0.0, -0.0001, 0.0)

    gpkg_files = []

    # Process multiple tiles
    for tile_idx in range(3):
        print(f"\nProcessing tile {tile_idx + 1}/3...")

        # Create synthetic predictions
        predictions = np.random.randint(0, 7, (256, 256), dtype=np.uint8)
        confidence = np.random.rand(256, 256) * 0.7 + 0.3

        # Export
        output_paths = exporter.export_predictions(
            predictions=predictions,
            transform=transform,
            output_name=f"tile_{tile_idx:02d}",
            confidence=confidence
        )

        if "geopackage" in output_paths:
            gpkg_files.append(output_paths["geopackage"])

    # Merge all GeoPackages
    if len(gpkg_files) > 1:
        print(f"\nMerging {len(gpkg_files)} GeoPackages...")
        merged_gpkg = exporter.create_merged_geopackage(
            gpkg_paths=gpkg_files,
            output_name="merged_tiles"
        )
        print(f"✓ Merged GeoPackage: {merged_gpkg}")

    return gpkg_files


def example_geojson_export():
    """
    Export example: Generate GeoJSON for web mapping applications.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: GeoJSON Export for Web Mapping")
    print("="*70)

    import geopandas as gpd
    from shapely.geometry import box

    output_dir = "outputs/gis_exports_web"
    exporter = GISExporter(
        output_dir=output_dir,
        crs=CRS.from_epsg(4326),
        min_polygon_area=50.0
    )

    # Create sample GeoDataFrame
    geometries = [
        box(0, 0, 1, 1),
        box(1.5, 0, 2.5, 1),
        box(0, 1.5, 1, 2.5),
    ]
    
    properties = [
        {
            "id": 1,
            "class": "building_rcc",
            "class_id": 1,
            "area_m2": 10000,
            "perimeter_m": 400,
            "avg_confidence": 0.95
        },
        {
            "id": 2,
            "class": "building_tiled",
            "class_id": 2,
            "area_m2": 5000,
            "perimeter_m": 280,
            "avg_confidence": 0.87
        },
        {
            "id": 3,
            "class": "waterbody",
            "class_id": 6,
            "area_m2": 50000,
            "perimeter_m": 900,
            "avg_confidence": 0.92
        }
    ]

    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=CRS.from_epsg(4326))

    # Export to GeoJSON
    for layer_name in ["buildings", "water"]:
        geojson_path = exporter.export_to_geojson(
            gdf, "sample_features", layer_name
        )
        print(f"✓ GeoJSON exported: {geojson_path}")


def example_feature_class_info():
    """
    Display feature class information.
    """
    print("\n" + "="*70)
    print("FEATURE CLASSES REFERENCE")
    print("="*70)

    print("\nAvailable feature classes:")
    print(f"{'ID':<5} {'Name':<20} {'Type':<15} {'Export':<10}")
    print("-" * 50)
    
    for class_id, info in FEATURE_CLASSES.items():
        print(f"{class_id:<5} {info['name']:<20} {info['type']:<15} {str(info['export']):<10}")


def example_custom_config():
    """
    Example with custom configuration.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Configuration")
    print("="*70)

    config = {
        "inference": {
            "min_building_area": 100.0,  # Larger minimum area
            "min_road_length": 20.0,      # Longer minimum length
        }
    }

    output_dir = "outputs/gis_exports_custom"
    exporter = GISExporter(
        output_dir=output_dir,
        crs=CRS.from_epsg(32643),  # UTM Zone 43N
        min_polygon_area=config["inference"]["min_building_area"],
        min_line_length=config["inference"]["min_road_length"],
        config=config
    )

    print(f"\nExporter configured with:")
    print(f"  Min polygon area: {exporter.min_polygon_area} m²")
    print(f"  Min line length: {exporter.min_line_length} m")
    print(f"  CRS: {exporter.crs}")
    print(f"  Output directory: {exporter.output_dir}")

    return exporter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GIS Export Pipeline Examples"
    )
    parser.add_argument(
        "--example",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Example to run (1-5)"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("GIS EXPORT PIPELINE EXAMPLES")
    print("="*70)

    try:
        if args.example == 1:
            example_basic_export()
        elif args.example == 2:
            example_batch_export()
        elif args.example == 3:
            example_geojson_export()
        elif args.example == 4:
            example_custom_config()
        elif args.example == 5:
            example_feature_class_info()

        print("\n" + "="*70)
        print("✓ Example completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
        print("\n✗ Example failed with error")

