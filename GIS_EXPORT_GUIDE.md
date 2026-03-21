# GIS Export Pipeline - Refactored Output System

## Overview

The refactored output pipeline generates geospatial data in **both Shapefile and GeoPackage (.gpkg)** formats. Each feature class (buildings, roads, waterbodies) is exported as:

1. **Individual Shapefiles** - One `.shp` file per feature class with complete attribute tables
2. **Combined GeoPackage** - Single `.gpkg` file containing all layers

## Key Features

✅ **Multi-Format Export**
- ESRI Shapefiles (.shp) for compatibility with GIS software
- GeoPackage (.gpkg) for modern, single-file distribution
- GeoJSON support for web mapping applications

✅ **Full Geometry Support**
- Preserves polygon geometries from raster predictions
- Maintains coordinate reference systems (CRS)
- Supports both projected and geographic coordinates

✅ **Comprehensive Attributes**
- Area calculations (m²)
- Perimeter/length measurements (m)
- Confidence scores from predictions
- Feature class identification
- Unique IDs for each feature

✅ **Quality Assurance**
- Morphological filtering of predictions
- Minimum area/length filtering
- Polygon validation and repair
- Export metadata tracking

✅ **QGIS & GIS Tool Compatibility**
- Standard CRS definitions (EPSG codes)
- OGC-compliant geometry
- Standard attribute naming conventions

## File Structure

```
outputs/
├── predictions/
│   ├── {output_name}_predictions.tif          # Raster predictions
│   ├── {output_name}_visualization.tif        # Colored visualization
│   ├── {output_name}_features.gpkg            # ✨ All layers combined
│   ├── {output_name}_building_rcc.shp         # Individual shapefiles
│   ├── {output_name}_building_rcc.shx
│   ├── {output_name}_building_rcc.dbf
│   ├── {output_name}_building_tiled.shp
│   ├── {output_name}_road.shp
│   ├── {output_name}_waterbody.shp
│   ├── {output_name}_export_metadata.json     # Export statistics
│   └── {output_name}_metadata.json            # Processing metadata
```

## Feature Classes

| ID | Class Name | Type | Export | Description |
|----|------------|------|--------|-------------|
| 0 | background | other | ✗ | Background/no-data |
| 1 | building_rcc | building | ✓ | RCC/concrete buildings |
| 2 | building_tiled | building | ✓ | Tiled roof buildings |
| 3 | building_tin | building | ✓ | Metal/tin buildings |
| 4 | building_others | building | ✓ | Other building types |
| 5 | road | infrastructure | ✓ | Roads and pathways |
| 6 | waterbody | water | ✓ | Water bodies |

## Usage Examples

### Basic Export

```python
from src.inference.gis_export import GISExporter
from rasterio.crs import CRS
from affine import Affine
import numpy as np

# Initialize exporter
exporter = GISExporter(
    output_dir="outputs/gis",
    crs=CRS.from_epsg(32643),  # UTM Zone 43N
    min_polygon_area=50.0,      # 50 m² minimum
    min_line_length=5.0         # 5 m minimum
)

# Export predictions
output_paths = exporter.export_predictions(
    predictions=predictions_array,    # Shape: (H, W), dtype: uint8
    transform=geotransform,           # Affine transform from rasterio
    output_name="tile_001",
    confidence=confidence_scores      # Optional: (H, W) array
)

# Access outputs
print(output_paths)
# {
#     'shapefile_building_rcc': Path('...building_rcc.shp'),
#     'shapefile_road': Path('...road.shp'),
#     'shapefile_waterbody': Path('...waterbody.shp'),
#     'geopackage': Path('...features.gpkg'),
#     'metadata': Path('...export_metadata.json')
# }
```

### Integration with FeatureExtractor

```python
from src.inference import FeatureExtractor

# Initialize extractor with config
extractor = FeatureExtractor(config, model_path, device="cpu")

# Process image - automatically exports to both formats
output_paths = extractor.extract_features(
    input_path="large_drone_image.tif",
    output_name="project_001"
)

# Output includes both Shapefiles and GeoPackage
```

### Batch Processing with Merge

```python
from src.inference.gis_export import GISExporter
from pathlib import Path

exporter = GISExporter(output_dir="outputs/batch")

# Process multiple tiles
gpkg_files = []
for tile_path in tile_paths:
    outputs = exporter.export_predictions(...)
    gpkg_files.append(outputs['geopackage'])

# Merge all into single GeoPackage
merged = exporter.create_merged_geopackage(
    gpkg_paths=gpkg_files,
    output_name="entire_area"
)
```

### Export to GeoJSON (Web Mapping)

```python
import geopandas as gpd

# Export individual layer as GeoJSON
geojson_path = exporter.export_to_geojson(
    gdf=buildings_gdf,
    output_name="project_001",
    class_name="building_rcc"
)

# Use in web mapping (Leaflet, Mapbox, etc.)
```

## Opening Outputs in QGIS

### Method 1: GeoPackage (Recommended)

1. Open QGIS
2. **Layer → Add Layer → Add Vector Layer**
3. Select **GeoPackage** source type
4. Browse to `{output_name}_features.gpkg`
5. Select which layers to add (buildings, roads, water)
6. Click **Add**

### Method 2: Shapefiles

1. Open QGIS
2. **Layer → Add Layer → Add Vector Layer**
3. Browse to individual `.shp` files
4. Each shapefile is a separate layer
5. Repeat for each class

### Method 3: Script Loading

```python
import geopandas as gpd

# Load GeoPackage layers
buildings = gpd.read_file("features.gpkg", layer="building_rcc")
roads = gpd.read_file("features.gpkg", layer="road")
water = gpd.read_file("features.gpkg", layer="waterbody")

# Combine layers
all_features = gpd.GeoDataFrame(
    pd.concat([buildings, roads, water], ignore_index=True)
)
```

## Attribute Tables

### Building Features

| Column | Type | Description |
|--------|------|-------------|
| id | integer | Unique feature ID |
| class | string | Class name (building_rcc, etc.) |
| class_id | integer | Numeric class ID (1-4) |
| area_m2 | float | Area in square meters |
| perimeter_m | float | Perimeter in meters |
| avg_confidence | float | Average confidence score (0-1) |
| num_vertices | integer | Number of polygon vertices |

### Road Features

| Column | Type | Description |
|--------|------|-------------|
| id | integer | Unique feature ID |
| class | string | "road" |
| class_id | integer | 5 |
| area_m2 | float | Area in square meters |
| perimeter_m | float | Length in meters |
| avg_confidence | float | Average confidence score |

### Water Features

| Column | Type | Description |
|--------|------|-------------|
| id | integer | Unique feature ID |
| class | string | "waterbody" |
| class_id | integer | 6 |
| area_m2 | float | Area in square meters |
| perimeter_m | float | Perimeter in meters |
| avg_confidence | float | Average confidence score |

## Configuration Options

```python
exporter = GISExporter(
    output_dir="outputs/gis",              # Output directory
    crs=CRS.from_epsg(32643),             # Coordinate Reference System
    min_polygon_area=50.0,                # Minimum polygon area (m²)
    min_line_length=5.0,                  # Minimum line length (m)
    config={                              # Optional: custom config
        "inference": {
            "min_building_area": 100.0,
            "min_road_length": 20.0,
        }
    }
)
```

## CRS (Coordinate Reference System)

### Common CRS Options

```python
from rasterio.crs import CRS

# Geographic (Latitude/Longitude)
crs = CRS.from_epsg(4326)  # WGS84 (Global)

# Projected (UTM - Universal Transverse Mercator)
crs = CRS.from_epsg(32643)  # UTM Zone 43N (India)
crs = CRS.from_epsg(32631)  # UTM Zone 31N (West Africa)

# From PROJ string
crs = CRS.from_proj4("+proj=utm +zone=43 +datum=WGS84")

# From WKT
crs = CRS.from_wkt("PROJCS[...]")

# From EPSG string
crs = CRS.from_string("EPSG:32643")
```

## Export Metadata

The `{output_name}_export_metadata.json` contains:

```json
{
  "timestamp": "2026-03-21T10:30:00",
  "output_name": "tile_001",
  "prediction_shape": [512, 512],
  "crs": "EPSG:32643",
  "output_files": {
    "shapefile_building_rcc": "outputs/tile_001_building_rcc.shp",
    "shapefile_road": "outputs/tile_001_road.shp",
    "geopackage": "outputs/tile_001_features.gpkg",
    "metadata": "outputs/tile_001_export_metadata.json"
  },
  "class_statistics": {
    "background": 245000,
    "building_rcc": 5000,
    "road": 3500,
    "waterbody": 1200
  },
  "export_settings": {
    "min_polygon_area": 50.0,
    "min_line_length": 5.0
  }
}
```

## Quality Checks

The pipeline automatically:

✓ Removes invalid/self-intersecting polygons
✓ Filters geometries below minimum size thresholds
✓ Validates CRS consistency
✓ Checks file integrity after export
✓ Logs detailed statistics for each class

### Run Validation

```python
validation_results = exporter.validate_exports()
for filename, is_valid in validation_results.items():
    status = "✓" if is_valid else "✗"
    print(f"{status} {filename}")
```

## Troubleshooting

### GeoPackage not opening in QGIS
- Check CRS is defined: `gdf.crs`
- Verify file exists: `Path(...).exists()`
- Try opening from QGIS UI instead of script

### Missing layers in GeoPackage
- Check export metadata for class statistics
- Verify min_polygon_area/min_line_length aren't too high
- Look for errors in console logs

### Shapefile character encoding issues
- GeoPackage is recommended (avoids encoding problems)
- Ensure output directory has write permissions

### Large file sizes
- GeoPackage is more efficient than multiple Shapefiles
- Use min_polygon_area to filter small features

## Performance Considerations

| Operation | Time | Notes |
|-----------|------|-------|
| Export 512×512 predictions | ~0.5s | Single class extraction |
| Create GeoPackage (all classes) | ~2s | File writing |
| Merge 10 GeoPackages | ~5s | I/O intensive |
| Validate exports | ~1s | File integrity check |

## References

- [QGIS Official Documentation](https://qgis.org/en/docs/)
- [GeoPackage Specification](http://www.geopackage.org/)
- [ESRI Shapefile Format](https://www.esri.com/content/dam/esrisites/sitecore/Home/Microsites/rest-apis/Files/pdfs/GIS%20Developers%20Handbook.pdf)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [GeoPandas Documentation](https://geopandas.org/)

