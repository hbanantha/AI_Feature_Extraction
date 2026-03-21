"""
GIS Export Pipeline - Validation & Testing Tools
=================================================
Comprehensive testing and validation scripts for exported GIS files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import json
import logging
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GISFileValidator:
    """Validate exported GIS files for quality and compatibility."""

    @staticmethod
    def validate_shapefile(shp_path):
        """Validate a single Shapefile."""
        try:
            gdf = gpd.read_file(shp_path)
            
            results = {
                "file": Path(shp_path).name,
                "valid": True,
                "features": len(gdf),
                "crs": str(gdf.crs),
                "geometry_types": list(gdf.geometry.type.unique()),
                "bounds": gdf.total_bounds.tolist(),
                "columns": list(gdf.columns)
            }
            
            # Check for invalid geometries
            invalid_count = (~gdf.geometry.is_valid).sum()
            results["invalid_geometries"] = invalid_count
            
            # Check for empty geometries
            empty_count = (gdf.geometry.is_empty).sum()
            results["empty_geometries"] = empty_count
            
            return results
            
        except Exception as e:
            return {
                "file": Path(shp_path).name,
                "valid": False,
                "error": str(e)
            }

    @staticmethod
    def validate_geopackage(gpkg_path):
        """Validate a GeoPackage file."""
        try:
            import fiona
            
            layers = fiona.listlayers(str(gpkg_path))
            results = {
                "file": Path(gpkg_path).name,
                "valid": True,
                "layers": layers,
                "layer_details": {}
            }
            
            # Validate each layer
            for layer_name in layers:
                try:
                    gdf = gpd.read_file(gpkg_path, layer=layer_name)
                    results["layer_details"][layer_name] = {
                        "features": len(gdf),
                        "crs": str(gdf.crs),
                        "geometry_types": list(gdf.geometry.type.unique()),
                        "bounds": gdf.total_bounds.tolist()
                    }
                except Exception as e:
                    results["layer_details"][layer_name] = {"error": str(e)}
            
            return results
            
        except Exception as e:
            return {
                "file": Path(gpkg_path).name,
                "valid": False,
                "error": str(e)
            }

    @staticmethod
    def validate_metadata(metadata_path):
        """Validate export metadata file."""
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                "file": Path(metadata_path).name,
                "valid": True,
                "timestamp": metadata.get("timestamp"),
                "output_name": metadata.get("output_name"),
                "prediction_shape": metadata.get("prediction_shape"),
                "crs": metadata.get("crs"),
                "class_statistics": metadata.get("class_statistics")
            }
            
        except Exception as e:
            return {
                "file": Path(metadata_path).name,
                "valid": False,
                "error": str(e)
            }


def load_and_inspect_geopackage(gpkg_path):
    """Load and inspect all layers in a GeoPackage."""
    print("\n" + "="*70)
    print("GEOPACKAGE INSPECTION")
    print("="*70)
    print(f"\nFile: {gpkg_path}\n")
    
    try:
        import fiona
        layers = fiona.listlayers(str(gpkg_path))
        print(f"Total layers: {len(layers)}\n")
        
        all_features = 0
        for layer_name in layers:
            gdf = gpd.read_file(gpkg_path, layer=layer_name)
            all_features += len(gdf)
            
            print(f"Layer: {layer_name}")
            print(f"  Features: {len(gdf)}")
            print(f"  CRS: {gdf.crs}")
            print(f"  Bounds: {gdf.total_bounds}")
            print(f"  Columns: {list(gdf.columns)}")
            
            if len(gdf) > 0:
                print(f"  Sample attributes:")
                sample = gdf[['class', 'area_m2', 'avg_confidence']].head(2)
                for idx, row in sample.iterrows():
                    print(f"    - {row['class']}: {row['area_m2']:.2f} m², confidence: {row['avg_confidence']:.3f}")
            print()
        
        print(f"Total features across all layers: {all_features}\n")
        
    except Exception as e:
        logger.error(f"Error inspecting GeoPackage: {e}")


def compare_shapefiles_vs_geopackage(output_dir):
    """Compare individual Shapefiles with GeoPackage."""
    print("\n" + "="*70)
    print("SHAPEFILE vs GEOPACKAGE COMPARISON")
    print("="*70)
    
    output_dir = Path(output_dir)
    
    # Find files
    shapefiles = list(output_dir.glob("*.shp"))
    geopackages = list(output_dir.glob("*.gpkg"))
    
    if not shapefiles or not geopackages:
        print("No shapefiles or GeoPackage found")
        return
    
    comparison_data = []
    
    # Count features in shapefiles
    shapefile_total = 0
    for shp in shapefiles:
        gdf = gpd.read_file(shp)
        class_name = shp.stem.split("_", 1)[1] if "_" in shp.stem else "unknown"
        shapefile_total += len(gdf)
        comparison_data.append({
            "Source": "Shapefile",
            "Class": class_name,
            "Features": len(gdf),
            "File": shp.name
        })
    
    # Count features in GeoPackage layers
    try:
        import fiona
        for gpkg in geopackages:
            layers = fiona.listlayers(str(gpkg))
            gpkg_total = 0
            
            for layer_name in layers:
                gdf = gpd.read_file(gpkg, layer=layer_name)
                gpkg_total += len(gdf)
                comparison_data.append({
                    "Source": "GeoPackage",
                    "Class": layer_name,
                    "Features": len(gdf),
                    "File": gpkg.name
                })
    except Exception as e:
        logger.warning(f"Could not inspect GeoPackage: {e}")
    
    print("\n" + tabulate(
        comparison_data,
        headers="keys",
        tablefmt="grid"
    ))
    
    print(f"\n✓ Shapefiles verified: {len(shapefiles)} files")
    print(f"✓ GeoPackage verified: {len(geopackages)} file(s)")


def generate_validation_report(output_dir):
    """Generate comprehensive validation report."""
    print("\n" + "="*70)
    print("GIS EXPORT VALIDATION REPORT")
    print("="*70)
    
    output_dir = Path(output_dir)
    validator = GISFileValidator()
    
    # Find all files
    shapefiles = list(output_dir.glob("*.shp"))
    geopackages = list(output_dir.glob("*.gpkg"))
    metadata_files = list(output_dir.glob("*_export_metadata.json"))
    
    print(f"\nFiles Found:")
    print(f"  Shapefiles: {len(shapefiles)}")
    print(f"  GeoPackages: {len(geopackages)}")
    print(f"  Metadata: {len(metadata_files)}")
    
    # Validate shapefiles
    if shapefiles:
        print("\n" + "-"*70)
        print("SHAPEFILE VALIDATION")
        print("-"*70)
        
        shapefile_results = []
        for shp in shapefiles:
            result = validator.validate_shapefile(shp)
            shapefile_results.append(result)
            
            status = "✓ VALID" if result.get("valid", False) else "✗ INVALID"
            print(f"\n{status}: {result['file']}")
            
            if result.get("valid"):
                print(f"  Features: {result['features']}")
                print(f"  CRS: {result['crs']}")
                print(f"  Geometry types: {result['geometry_types']}")
                print(f"  Invalid geometries: {result['invalid_geometries']}")
                print(f"  Empty geometries: {result['empty_geometries']}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Validate GeoPackage
    if geopackages:
        print("\n" + "-"*70)
        print("GEOPACKAGE VALIDATION")
        print("-"*70)
        
        for gpkg in geopackages:
            result = validator.validate_geopackage(gpkg)
            
            status = "✓ VALID" if result.get("valid", False) else "✗ INVALID"
            print(f"\n{status}: {result['file']}")
            
            if result.get("valid"):
                print(f"  Layers: {len(result['layers'])}")
                print(f"  Layer names: {result['layers']}")
                
                for layer_name, details in result['layer_details'].items():
                    if 'error' not in details:
                        print(f"\n  Layer: {layer_name}")
                        print(f"    Features: {details['features']}")
                        print(f"    CRS: {details['crs']}")
                        print(f"    Geometry types: {details['geometry_types']}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Validate metadata
    if metadata_files:
        print("\n" + "-"*70)
        print("METADATA VALIDATION")
        print("-"*70)
        
        for meta in metadata_files:
            result = validator.validate_metadata(meta)
            
            status = "✓ VALID" if result.get("valid", False) else "✗ INVALID"
            print(f"\n{status}: {result['file']}")
            
            if result.get("valid"):
                print(f"  Output: {result['output_name']}")
                print(f"  Timestamp: {result['timestamp']}")
                print(f"  Prediction shape: {result['prediction_shape']}")
                print(f"  CRS: {result['crs']}")
                print(f"  Class statistics:")
                for class_name, count in result['class_statistics'].items():
                    print(f"    - {class_name}: {count} pixels")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


def check_compatibility():
    """Check system compatibility for GIS operations."""
    print("\n" + "="*70)
    print("SYSTEM COMPATIBILITY CHECK")
    print("="*70)
    
    compatibility_status = []
    
    # Check required packages
    packages = {
        "geopandas": "GeoPandas (vector data)",
        "rasterio": "Rasterio (raster I/O)",
        "shapely": "Shapely (geometry)",
        "numpy": "NumPy (arrays)",
        "cv2": "OpenCV (image processing)"
    }
    
    print("\nRequired Packages:")
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {description}")
            compatibility_status.append((package, True))
        except ImportError:
            print(f"  ✗ {description}")
            compatibility_status.append((package, False))
    
    # Check optional packages
    optional_packages = {
        "fiona": "Fiona (GeoPackage layers)",
    }
    
    print("\nOptional Packages:")
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {description}")
        except ImportError:
            print(f"  ✗ {description} - Install with: pip install {package}")
    
    # Summary
    required_ok = all(status for _, status in compatibility_status)
    print("\n" + "="*70)
    if required_ok:
        print("✓ System is compatible - all required packages installed")
    else:
        print("✗ Missing required packages - install with: pip install -r requirements.txt")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GIS Export Validation Tools")
    parser.add_argument("--output-dir", default="outputs/gis_exports", 
                       help="Output directory to validate")
    parser.add_argument("--check-compat", action="store_true",
                       help="Check system compatibility")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation report")
    parser.add_argument("--inspect-gpkg", 
                       help="Inspect specific GeoPackage file")
    parser.add_argument("--compare", action="store_true",
                       help="Compare Shapefiles vs GeoPackage")
    
    args = parser.parse_args()
    
    # Default: run all checks
    if not any([args.check_compat, args.validate, args.inspect_gpkg, args.compare]):
        args.check_compat = True
        args.validate = True
        args.compare = True
    
    if args.check_compat:
        check_compatibility()
    
    if args.validate:
        generate_validation_report(args.output_dir)
    
    if args.compare:
        compare_shapefiles_vs_geopackage(args.output_dir)
    
    if args.inspect_gpkg:
        load_and_inspect_geopackage(args.inspect_gpkg)

