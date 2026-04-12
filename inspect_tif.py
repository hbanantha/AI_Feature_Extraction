import rasterio
from rasterio.errors import RasterioIOError
from pathlib import Path

# Path to the raw data directory
raw_dir = Path(r"G:\MTech Cyber Security\IIT_Tirupathi_AI_ML_Hackathon\Dataset\FINAL_DATASET\1TD_Set1")

print("\n=== GeoTIFF Inspection Report ===\n")

# Supported raster file extensions
supported_extensions = (".tif", ".tiff")

for file_path in raw_dir.glob("*"):
    if file_path.suffix.lower() not in supported_extensions:
        continue

    print(f"Checking: {file_path.name}")

    try:
        with rasterio.open(file_path) as src:
            print(f"  Status        : Readable")
            print(f"  Driver        : {src.driver}")
            print(f"  Size          : {src.width} x {src.height}")
            print(f"  Bands         : {src.count}")
            print(f"  Data Type     : {src.dtypes}")
            print(f"  CRS           : {src.crs}")
            print(f"  Compression   : {src.compression}")
            print(f"  Bounds        : {src.bounds}")
            print(f"  ColorInterp   : {src.colorinterp}")

            # Determine image type
            if src.count == 4:
                print("  Image Type    : RGBA (Alpha channel present)")
            elif src.count == 3:
                print("  Image Type    : RGB")
            else:
                print("  Image Type    : Multispectral/Other")

    except RasterioIOError as e:
        print(f"  Status        : Error reading file")
        print(f"  Error         : {e}")

    except Exception as e:
        print(f"  Status        : Unexpected error")
        print(f"  Error         : {e}")

    print("-" * 60)

print("\nInspection completed.\n")