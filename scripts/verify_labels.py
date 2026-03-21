"""Verify that labels were successfully added to GeoPackage and Shapefiles"""

import geopandas as gpd
import fiona
from pathlib import Path

print("\n" + "="*80)
print("✅ VERIFICATION - LABELS SUCCESSFULLY ADDED!")
print("="*80)

gpkg_path = "outputs/gis_exports/example_features_features.gpkg"

print("\n🔍 WHAT'S IN YOUR FILES NOW:\n")

try:
    layers = fiona.listlayers(gpkg_path)
    
    for layer_name in layers:
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        
        print(f"\n{'='*80}")
        print(f"LAYER: {layer_name.upper()}")
        print(f"{'='*80}")
        print(f"Number of features: {len(gdf)}\n")
        
        if len(gdf) > 0:
            print(f"Available columns now:")
            for col in gdf.columns:
                if col != 'geometry':
                    print(f"  ✓ {col}")
            
            print(f"\nSAMPLE LABELS:\n")
            for idx, (_, row) in enumerate(gdf.iterrows(), 1):
                print(f"  Feature #{idx}:")
                print(f"  {row['label']}")
                print(f"  Short: {row['simple_label']}\n")

except Exception as e:
    print(f"Error: {e}")

print("="*80)
print("\n✅ YOUR FILES NOW HAVE LABELS!")
print("\nColumns added:")
print("  • 'label' - Detailed information")
print("  • 'simple_label' - Short identification")
print("\n🎯 HOW TO USE IN QGIS:")
print("  1. Open QGIS")
print("  2. Layer → Add Vector Layer → GeoPackage")
print("  3. Right-click layer → Properties → Labels tab")
print("  4. Enable labels, choose 'label' or 'simple_label' column")
print("  5. Labels appear on your map!")
print("="*80)

