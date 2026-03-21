# GIS Export Quick Reference Guide

## 🚀 Getting Started

### Step 1: Generate Exports
```python
from src.inference import FeatureExtractor

# Initialize
extractor = FeatureExtractor(config, model_path, device="cpu")

# Process image
outputs = extractor.extract_features("input_image.tif")

# Automatically generates both Shapefiles and GeoPackage!
```

## 📁 Output Files

### Shapefiles (Individual by Class)
- `{name}_building_rcc.shp` - RCC/concrete buildings
- `{name}_building_tiled.shp` - Tiled/slate buildings
- `{name}_building_tin.shp` - Metal/tin buildings
- `{name}_building_others.shp` - Other building types
- `{name}_road.shp` - Roads and pathways
- `{name}_waterbody.shp` - Water bodies

### GeoPackage (All Layers)
- `{name}_features.gpkg` - **Single file with all layers** ⭐

### Supporting Files
- `{name}_export_metadata.json` - Statistics and settings
- `{name}_predictions.tif` - Raster predictions
- `{name}_visualization.tif` - Colored visualization

## 🖥️ Opening in QGIS

### Method 1: GeoPackage (Recommended)
1. **Layer → Add Layer → Add Vector Layer**
2. Select **GeoPackage** as source type
3. Browse to `{name}_features.gpkg`
4. Choose which layers to add (or add all)
5. Click **Add**

### Method 2: Individual Shapefiles
1. **Layer → Add Layer → Add Vector Layer**
2. Select **File** source type
3. Browse to `.shp` files
4. Repeat for each file

### Method 3: Drag & Drop
Simply drag `.gpkg` or `.shp` files into QGIS canvas

## 📊 Feature Attributes

All exported features include:

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | Integer | Unique feature ID |
| `class` | String | Feature class name |
| `class_id` | Integer | Class numeric ID |
| `area_m2` | Float | Area in square meters |
| `perimeter_m` | Float | Perimeter/length in meters |
| `avg_confidence` | Float | Model confidence (0-1) |
| `num_vertices` | Integer | Number of polygon vertices |

## 🐍 Loading in Python

### Load GeoPackage
```python
import geopandas as gpd

# Load specific layer
buildings = gpd.read_file("features.gpkg", layer="building_rcc")

# Load all layers
for layer in ["building_rcc", "building_tiled", "road", "waterbody"]:
    gdf = gpd.read_file("features.gpkg", layer=layer)
    print(f"{layer}: {len(gdf)} features")
```

### Load Shapefile
```python
import geopandas as gpd

gdf = gpd.read_file("example_features_building_rcc.shp")
print(gdf.head())
print(gdf.info())
```

### Query Features
```python
# Get all buildings > 1000 m²
large_buildings = buildings[buildings['area_m2'] > 1000]

# Get high-confidence features
confident = buildings[buildings['avg_confidence'] > 0.9]

# Get features by bounds
bounds = buildings.total_bounds
print(f"Total area covered: {bounds}")
```

## 🔍 Validation & Testing

### Check System Compatibility
```bash
python scripts/validate_gis_exports.py --check-compat
```

### Validate Exports
```bash
python scripts/validate_gis_exports.py --validate
```

### Compare Shapefiles vs GeoPackage
```bash
python scripts/validate_gis_exports.py --compare
```

### Inspect GeoPackage Contents
```bash
python scripts/validate_gis_exports.py --inspect-gpkg outputs/features.gpkg
```

## ⚙️ Configuration

### Custom Export Settings
```python
from src.inference.gis_export import GISExporter
from rasterio.crs import CRS

exporter = GISExporter(
    output_dir="outputs/gis",
    crs=CRS.from_epsg(32643),      # UTM Zone 43N
    min_polygon_area=50.0,          # 50 m² minimum
    min_line_length=5.0,            # 5 m minimum
    config={...}                    # Optional config
)
```

### Common CRS Codes
```python
from rasterio.crs import CRS

# WGS84 (Global coordinates)
CRS.from_epsg(4326)

# UTM Zone 43N (India)
CRS.from_epsg(32643)

# UTM Zone 31N (West Africa)
CRS.from_epsg(32631)

# Indian State Plane Coordinate System
CRS.from_epsg(32643)
```

## 📈 Feature Statistics

Check what was exported:
```python
import json

with open("example_features_export_metadata.json", 'r') as f:
    metadata = json.load(f)

print("Class Statistics:")
for class_name, count in metadata['class_statistics'].items():
    print(f"  {class_name}: {count} pixels")
```

## 🎨 Styling in QGIS

### Color by Class
1. Right-click layer → **Properties**
2. Go to **Symbology** tab
3. Select **Categorized**
4. Choose `class` column
5. Click **Classify**

### Color by Confidence
1. Right-click layer → **Properties**
2. Go to **Symbology** tab
3. Select **Graduated**
4. Choose `avg_confidence` column
5. Set ramp (e.g., low=red, high=green)

### Filter by Size
1. Right-click layer → **Properties**
2. Go to **General** tab
3. Set **Query Builder**
4. Example: `area_m2 > 100`

## 📋 File Format Comparison

| Feature | Shapefile | GeoPackage |
|---------|-----------|-----------|
| Single File | ✗ (3 files) | ✓ |
| File Size | Large | Smaller |
| Layers | 1 per file | Multiple |
| CRS Handling | Basic | Excellent |
| QGIS Support | ✓ | ✓ |
| ArcGIS Support | ✓ | ✓ |
| Web GIS | Limited | ✓ |
| Recommended | Legacy workflows | New projects |

## 🐛 Troubleshooting

### "File not found"
- Check output directory path
- Verify export completed successfully
- Check logs for errors

### "Invalid CRS"
- Source image must have CRS defined
- GeoPackage will auto-detect or use WGS84

### "Empty geometries"
- Check min_polygon_area setting
- Verify prediction contains class pixels
- Inspect metadata statistics

### "QGIS won't open file"
- Try opening from QGIS UI instead of shell
- Ensure file extension is correct
- Check file permissions

## 📚 Examples

### Run All Examples
```bash
python scripts/gis_export_examples.py --example 1  # Basic export
python scripts/gis_export_examples.py --example 2  # Batch processing
python scripts/gis_export_examples.py --example 3  # GeoJSON export
python scripts/gis_export_examples.py --example 4  # Custom config
python scripts/gis_export_examples.py --example 5  # Feature classes
```

## ✅ Verification Checklist

After export, verify:
- [ ] Shapefiles created (5 minimum)
- [ ] GeoPackage file exists
- [ ] All feature counts > 0
- [ ] CRS is correct (EPSG code)
- [ ] Can open in QGIS
- [ ] Attribute tables visible
- [ ] Geometries are valid

## 🔗 Related Documentation

- **GIS_EXPORT_GUIDE.md** - Complete technical documentation
- **GIS_EXPORT_REFACTORING_SUMMARY.md** - Architecture and changes
- **src/inference/gis_export.py** - Source code with docstrings

## 🤝 Support

For issues or questions:
1. Check validation output: `python scripts/validate_gis_exports.py`
2. Review logs for error messages
3. Verify input data quality
4. Test with example script first

## 🎯 Best Practices

✅ **DO**
- Use GeoPackage for distribution (single file)
- Validate exports after generation
- Check CRS matches your workflow
- Review metadata for statistics

❌ **DON'T**
- Edit exported files outside GIS tools
- Assume CRS is correct without checking
- Export with very high min_polygon_area
- Store in old Shapefile format only

---

**Version:** 1.0 | **Date:** March 2026 | **Status:** Production Ready ✓

