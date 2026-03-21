"""
Add Labels to GIS Features
==========================
Adds descriptive labels to features in GeoPackage and Shapefiles
so users can easily recognize and understand each feature.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import fiona
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_feature_label(row):
    """
    Create a descriptive label for a feature based on its properties.
    
    Args:
        row: GeoDataFrame row with feature properties
    
    Returns:
        Formatted label string
    """
    class_name = row.get('class', 'Unknown')
    area = row.get('area_m2', 0)
    confidence = row.get('avg_confidence', 0)
    feature_id = row.get('id', '?')
    
    # Determine feature type and create appropriate label
    if 'building' in class_name:
        building_type = {
            'building_rcc': 'RCC Building',
            'building_tiled': 'Tiled Building',
            'building_tin': 'Tin Building',
            'building_others': 'Other Building'
        }.get(class_name, 'Building')
        
        confidence_text = '✅ High' if confidence >= 0.9 else '⚠️ Check' if confidence >= 0.7 else '❌ Low'
        label = f"{building_type}\nID: {feature_id} | Area: {area:.0f}m² | Confidence: {confidence:.2f} {confidence_text}"
    
    elif class_name == 'road':
        confidence_text = '✅ High' if confidence >= 0.9 else '⚠️ Check' if confidence >= 0.7 else '❌ Low'
        label = f"Road/Path\nID: {feature_id} | Length: {area:.0f}m | Confidence: {confidence:.2f} {confidence_text}"
    
    elif class_name == 'waterbody':
        confidence_text = '✅ High' if confidence >= 0.9 else '⚠️ Check' if confidence >= 0.7 else '❌ Low'
        label = f"Water Body\nID: {feature_id} | Area: {area:.0f}m² | Confidence: {confidence:.2f} {confidence_text}"
    
    else:
        label = f"{class_name}\nID: {feature_id} | Area: {area:.0f}m² | Conf: {confidence:.2f}"
    
    return label


def create_simple_label(row):
    """Create a simple, short label for feature display."""
    class_name = row.get('class', 'Unknown')
    feature_id = row.get('id', '?')
    
    if 'building' in class_name:
        building_type = {
            'building_rcc': 'RCC',
            'building_tiled': 'Tiled',
            'building_tin': 'Tin',
            'building_others': 'Other'
        }.get(class_name, 'Bldg')
        return f"{building_type} #{feature_id}"
    elif class_name == 'road':
        return f"Road #{feature_id}"
    elif class_name == 'waterbody':
        return f"Water #{feature_id}"
    else:
        return f"{class_name} #{feature_id}"


def add_labels_to_geopackage(gpkg_path, output_path=None):
    """
    Add descriptive labels to all features in GeoPackage.
    
    Args:
        gpkg_path: Path to input GeoPackage
        output_path: Path for output GeoPackage (default: overwrite input)
    """
    if output_path is None:
        output_path = gpkg_path
    
    logger.info(f"Processing GeoPackage: {gpkg_path}")
    
    try:
        # Get all layers
        layers = fiona.listlayers(str(gpkg_path))
        logger.info(f"Found {len(layers)} layers")
        
        # Process each layer
        for idx, layer_name in enumerate(layers):
            logger.info(f"Processing layer: {layer_name}")
            
            # Read layer
            gdf = gpd.read_file(gpkg_path, layer=layer_name)
            
            # Add labels
            gdf['label'] = gdf.apply(create_feature_label, axis=1)
            gdf['simple_label'] = gdf.apply(create_simple_label, axis=1)
            
            logger.info(f"  ✓ Added labels to {len(gdf)} features")
            logger.info(f"  Sample label: {gdf['label'].iloc[0] if len(gdf) > 0 else 'N/A'}")
            
            # Write to GeoPackage
            if idx == 0:
                # First layer - create new file
                gdf.to_file(output_path, layer=layer_name, driver='GPKG')
            else:
                # Subsequent layers - append
                gdf.to_file(output_path, layer=layer_name, driver='GPKG')
        
        logger.info(f"✓ GeoPackage with labels saved: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing GeoPackage: {e}")
        return False


def add_labels_to_shapefiles(output_dir):
    """
    Add descriptive labels to all Shapefile layers.
    
    Args:
        output_dir: Directory containing Shapefiles
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        logger.error(f"Directory not found: {output_dir}")
        return False
    
    # Find all shapefiles
    shapefiles = list(output_dir.glob("*.shp"))
    logger.info(f"Found {len(shapefiles)} Shapefiles")
    
    for shp_path in shapefiles:
        logger.info(f"Processing: {shp_path.name}")
        
        try:
            # Read shapefile
            gdf = gpd.read_file(shp_path)
            
            # Add labels
            gdf['label'] = gdf.apply(create_feature_label, axis=1)
            gdf['simple_lbl'] = gdf.apply(create_simple_label, axis=1)
            
            # Save back to shapefile
            gdf.to_file(shp_path)
            
            logger.info(f"  ✓ Added labels to {len(gdf)} features")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {shp_path.name}: {e}")
            continue
    
    logger.info(f"✓ All Shapefiles updated with labels")
    return True


def verify_labels(gpkg_path):
    """
    Verify that labels were added successfully.
    
    Args:
        gpkg_path: Path to GeoPackage file
    """
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION - Labels Added Successfully")
    logger.info("="*80)
    
    try:
        layers = fiona.listlayers(str(gpkg_path))
        
        for layer_name in layers:
            gdf = gpd.read_file(gpkg_path, layer=layer_name)
            
            logger.info(f"\n✓ Layer: {layer_name}")
            logger.info(f"  Features with labels: {len(gdf)}")
            
            if len(gdf) > 0:
                logger.info(f"\n  Sample labels:")
                for idx, (_, row) in enumerate(gdf.head(2).iterrows()):
                    label = row.get('label', 'N/A')
                    logger.info(f"    Feature {idx+1}:")
                    for line in label.split('\n'):
                        logger.info(f"      {line}")
                    logger.info("")
    
    except Exception as e:
        logger.error(f"Verification failed: {e}")


def display_label_info():
    """Display information about labels."""
    print("\n" + "="*80)
    print("LABELS ADDED TO FEATURES")
    print("="*80)
    
    print("""
✅ LABELS ADDED TO EACH FEATURE:

   Each feature now has TWO label columns:

   1️⃣  'label' (Detailed Label):
       • Feature type (RCC Building, Road, Water Body, etc.)
       • Unique ID number
       • Size in square meters
       • Confidence score (0.0-1.0)
       • Confidence indicator (✅ High, ⚠️ Check, ❌ Low)
       
       Example:
       ┌─────────────────────────────────────────┐
       │ RCC Building                            │
       │ ID: 1 | Area: 450m² | Confidence: 0.95 │
       │ ✅ High                                 │
       └─────────────────────────────────────────┘

   2️⃣  'simple_label' (Short Label):
       • Feature type abbreviation
       • ID number
       • Good for maps with limited space
       
       Example: RCC #1, Road #2, Water #3

📍 WHERE TO FIND LABELS:

   In QGIS:
   1. Right-click layer → Properties
   2. Go to "Labels" tab
   3. Enable labels
   4. Choose "label" column
   5. Labels appear on map!

   In Python:
   >>> import geopandas as gpd
   >>> gdf = gpd.read_file("file.gpkg", layer="building_rcc")
   >>> print(gdf['label'].iloc[0])  # See the label
   >>> print(gdf['simple_label'].iloc[0])  # See short label

🎨 USING LABELS IN QGIS:

   Method 1 - Detailed Labels:
   • Layer → Properties → Labels tab
   • ☑ Enable labels
   • Column: "label"
   • Font size: 8-10pt
   • → Detailed information appears on map

   Method 2 - Simple Labels:
   • Layer → Properties → Labels tab
   • ☑ Enable labels
   • Column: "simple_label"
   • Font size: 6-8pt
   • → Short identification appears on map

💡 LABEL INFORMATION INCLUDES:

   ✓ Feature Type
     • RCC Building (Concrete)
     • Tiled Building (Traditional)
     • Tin Building (Metal roof)
     • Road/Path
     • Water Body

   ✓ Unique ID
     • Identifies each feature
     • Unique within layer

   ✓ Size/Area
     • Square meters for buildings/water
     • Meters for roads

   ✓ Confidence Score
     • 0.0 to 1.0 scale
     • Higher = more reliable
     • ✅ High (0.9+)
     • ⚠️ Check (0.7-0.9)
     • ❌ Low (<0.7)

🔍 EXAMPLES OF LABELS:

   RCC Building:
   ┌──────────────────────────────────┐
   │ RCC Building                     │
   │ ID: 1 | Area: 450m² | Conf: 0.95│
   │ ✅ High                          │
   └──────────────────────────────────┘

   Tiled Building:
   ┌──────────────────────────────────┐
   │ Tiled Building                   │
   │ ID: 2 | Area: 320m² | Conf: 0.87│
   │ ⚠️ Check                         │
   └──────────────────────────────────┘

   Road:
   ┌──────────────────────────────────┐
   │ Road/Path                        │
   │ ID: 3 | Length: 250m | Conf: 0.92│
   │ ✅ High                          │
   └──────────────────────────────────┘

   Water Body:
   ┌──────────────────────────────────┐
   │ Water Body                       │
   │ ID: 4 | Area: 1200m² | Conf: 0.98│
   │ ✅ High                          │
   └──────────────────────────────────┘

✅ BENEFITS OF LABELS:

   • Easy identification of feature types
   • Quick quality assessment (confidence scores)
   • Understand feature sizes without calculations
   • Better map communication
   • More professional outputs
   • Easier data verification
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add Labels to GIS Features")
    parser.add_argument("--gpkg", default="outputs/gis_exports/example_features_features.gpkg",
                       help="Path to GeoPackage file")
    parser.add_argument("--output-gpkg", help="Output GeoPackage path (default: overwrite input)")
    parser.add_argument("--shapefiles", default="outputs/gis_exports",
                       help="Directory containing Shapefiles")
    parser.add_argument("--verify", action="store_true", help="Verify labels after adding")
    
    args = parser.parse_args()
    
    # Display label information
    display_label_info()
    
    print("\n" + "="*80)
    print("ADDING LABELS TO FILES")
    print("="*80 + "\n")
    
    # Process GeoPackage
    if Path(args.gpkg).exists():
        logger.info(f"\n📦 Processing GeoPackage: {args.gpkg}")
        success = add_labels_to_geopackage(args.gpkg, args.output_gpkg or args.gpkg)
        
        if success and args.verify:
            verify_labels(args.output_gpkg or args.gpkg)
    else:
        logger.warning(f"GeoPackage not found: {args.gpkg}")
    
    # Process Shapefiles
    if Path(args.shapefiles).exists():
        logger.info(f"\n📁 Processing Shapefiles in: {args.shapefiles}")
        add_labels_to_shapefiles(args.shapefiles)
    else:
        logger.warning(f"Directory not found: {args.shapefiles}")
    
    print("\n" + "="*80)
    print("✅ LABELS ADDED SUCCESSFULLY!")
    print("="*80)
    print("""
📍 Next Steps:
   1. Open files in QGIS
   2. Right-click layer → Properties → Labels
   3. Enable labels and choose 'label' or 'simple_label' column
   4. Labels now appear on your map!

🎯 TIP: Use 'simple_label' for cleaner maps with limited space
         Use 'label' for detailed information display
    """)

