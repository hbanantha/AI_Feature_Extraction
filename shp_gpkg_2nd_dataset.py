import geopandas as gpd
import os
import fiona
import pandas as pd

# -------------------------------
# INPUT AND OUTPUT PATHS
# -------------------------------
input_path = "input_shp/shp-file"  # Folder or dataset containing shapefiles
output_dir = "output_gpkg"
os.makedirs(output_dir, exist_ok=True)

# Match CRS with your orthophoto raster
TARGET_CRS = "EPSG:32644"

# -------------------------------
# GET ALL LAYERS
# -------------------------------
layers = fiona.listlayers(input_path)
print("Available layers:", layers)


# -------------------------------
# FUNCTION: CLEAN INVALID FIELDS
# -------------------------------
def clean_fields(gdf):
    """
    Removes problematic fields that cause errors in GeoPackage export.
    """
    invalid_fields = [
        "Shape_Area", "Shape_Length",
        "SHAPE_Area", "SHAPE_Length",
        "shape_area", "shape_length"
    ]

    cols_to_drop = [col for col in invalid_fields if col in gdf.columns]
    if cols_to_drop:
        print(f"Dropping invalid fields: {cols_to_drop}")
        gdf = gdf.drop(columns=cols_to_drop)

    return gdf


# -------------------------------
# FUNCTION: ENSURE VALID GEOMETRY
# -------------------------------
def ensure_valid_geometry(gdf):
    """
    Fixes invalid geometries and removes null geometries.
    """
    gdf = gdf[gdf.geometry.notnull()]
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


# -------------------------------
# FUNCTION: CLASSIFICATION
# -------------------------------
def assign_classes(row):
    cols = row.index

    # -----------------------
    # BUILDINGS
    # -----------------------
    if "Roof_type" in cols and pd.notna(row["Roof_type"]):
        rt = row["Roof_type"]
        if rt in [1, 2, 3, 4]:
            return (
                {1: 1, 2: 2, 3: 3, 4: 4}[rt],
                {1: "RCC", 2: "Tiled", 3: "Tin", 4: "Others"}[rt],
                "buildings"
            )

    # -----------------------
    # ROADS
    # -----------------------
    if "Road_type" in cols and pd.notna(row["Road_type"]):
        rt = row["Road_type"]
        if rt in [3, 5, 6]:
            return 5, {3: "Tar", 5: "Concrete", 6: "Mud"}[rt], "roads"

    # -----------------------
    # BRIDGE (treated as road)
    # -----------------------
    if "Bridge_typ" in cols and pd.notna(row["Bridge_typ"]):
        return 5, "Bridge", "roads"

    # -----------------------
    # WATER (polygon + line)
    # -----------------------
    if "Water_Body" in cols and pd.notna(row["Water_Body"]):
        wb = row["Water_Body"]
        wb_map = {
            1: "River",
            2: "Canal",
            5: "Pond/Lake",
            6: "Tank",
            10: "Borewell",
            8: "Drainage"
        }
        if wb in wb_map:
            layer = "drainage" if wb == 8 else "water"
            return 6, wb_map[wb], layer

    # -----------------------
    # WATER POINT (wells etc.)
    # -----------------------
    if "Water_Bodi" in cols and pd.notna(row["Water_Bodi"]):
        wb = row["Water_Bodi"]
        if wb == 3:
            return None, "Well", "assets"

    # -----------------------
    # UTILITIES
    # -----------------------
    if "Utility_Ty" in cols and pd.notna(row["Utility_Ty"]):
        ut = row["Utility_Ty"]
        if ut in [1, 2, 11]:
            return None, {1: "Pole", 2: "Tower", 11: "Station"}[ut], "assets"

    return None, None, None


# -------------------------------
# PROCESS EACH LAYER
# -------------------------------
all_data = []

for layer in layers:
    try:
        gdf = gpd.read_file(input_path, layer=layer)

        if gdf.empty:
            print(f"Skipping empty layer: {layer}")
            continue

        if len(gdf.columns) <= 1:
            print(f"Skipping no-attribute layer: {layer}")
            continue

        print(f"Processing layer: {layer}")

        # -------------------------------
        # CRS HANDLING
        # -------------------------------
        if gdf.crs is None:
            print(f"⚠️ CRS missing in {layer}. Assigning EPSG:4326.")
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)

        if gdf.crs.to_string() != TARGET_CRS:
            print(f"Reprojecting {layer} from {gdf.crs} to {TARGET_CRS}")
            gdf = gdf.to_crs(TARGET_CRS)

        # Clean invalid fields
        gdf = clean_fields(gdf)

        # Fix invalid geometries
        gdf = ensure_valid_geometry(gdf)

        # Apply classification
        gdf[["class_id", "subclass", "layer_name"]] = gdf.apply(
            assign_classes,
            axis=1,
            result_type="expand"
        )

        # Keep only valid rows
        gdf = gdf[gdf["layer_name"].notnull()]

        if not gdf.empty:
            all_data.append(gdf)

    except Exception as e:
        print(f"Error reading layer {layer}: {e}")


# -------------------------------
# MERGE ALL VALID DATA
# -------------------------------
if not all_data:
    raise ValueError("No valid data found!")

merged_gdf = gpd.GeoDataFrame(
    pd.concat(all_data, ignore_index=True),
    crs=TARGET_CRS
)

print(f"✅ Merged CRS: {merged_gdf.crs}")


# -------------------------------
# CHECK REQUIRED COLUMN
# -------------------------------
if "Village_Na" not in merged_gdf.columns:
    raise ValueError("Column 'Village_Na' not found in the dataset.")


# -------------------------------
# SPLIT BY VILLAGE AND SAVE
# -------------------------------
villages = merged_gdf["Village_Na"].dropna().unique()

for village in villages:
    village_gdf = merged_gdf[merged_gdf["Village_Na"] == village]

    output_path = os.path.join(output_dir, f"{village}.gpkg")
    print(f"Saving village: {village}")

    grouped = {
        "buildings": village_gdf[village_gdf["layer_name"] == "buildings"],
        "roads": village_gdf[village_gdf["layer_name"] == "roads"],
        "water": village_gdf[village_gdf["layer_name"] == "water"],
        "drainage": village_gdf[village_gdf["layer_name"] == "drainage"],
        "assets": village_gdf[village_gdf["layer_name"] == "assets"],
    }

    for layer_name, layer_gdf in grouped.items():
        if not layer_gdf.empty:
            layer_gdf = layer_gdf.reset_index(drop=True)

            layer_gdf.to_file(
                output_path,
                layer=layer_name,
                driver="GPKG",
                engine="pyogrio"
            )

print("✅ Done!")