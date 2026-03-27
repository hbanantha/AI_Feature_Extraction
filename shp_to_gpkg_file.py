import geopandas as gpd
import os
import fiona
import pandas as pd

input_path = "input_shp/all_villages_shp"  # folder or dataset
output_dir = "output_gpkg"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# GET ALL LAYERS
# -------------------------------
layers = fiona.listlayers(input_path)

print("Available layers:", layers)

# -------------------------------
# FUNCTION: CLASSIFICATION
# -------------------------------
def assign_classes(row):
    cols = row.index

    # -----------------------
    # BUILDINGS
    # -----------------------
    if "Roof_type" in cols and row["Roof_type"] is not None:
        rt = row["Roof_type"]
        if rt in [1, 2, 3, 4]:
            return (
                {1:1, 2:2, 3:3, 4:4}[rt],
                {1:"RCC", 2:"Tiled", 3:"Tin", 4:"Others"}[rt],
                "buildings"
            )

    # -----------------------
    # ROADS
    # -----------------------
    if "Road_type" in cols and row["Road_type"] is not None:
        rt = row["Road_type"]
        if rt in [3, 5, 6]:
            return 5, {3:"Tar", 5:"Concrete", 6:"Mud"}[rt], "roads"

    # -----------------------
    # BRIDGE (treat as road)
    # -----------------------
    if "Bridge_typ" in cols and row["Bridge_typ"] is not None:
        return 5, "Bridge", "roads"

    # -----------------------
    # WATER (polygon + line)
    # -----------------------
    if "Water_Body" in cols and row["Water_Body"] is not None:
        wb = row["Water_Body"]
        wb_map = {
            1:"River",
            2:"Canal",
            5:"Pond/Lake",
            6:"Tank",
            10:"Borewell",
            8:"Drainage"
        }
        if wb in wb_map:
            layer = "drainage" if wb == 8 else "water"
            return 6, wb_map[wb], layer

    # -----------------------
    # WATER POINT (wells etc.)
    # -----------------------
    if "Water_Bodi" in cols and row["Water_Bodi"] is not None:
        wb = row["Water_Bodi"]
        if wb == 3:
            return None, "Well", "assets"

    # -----------------------
    # UTILITIES
    # -----------------------
    if "Utility_Ty" in cols and row["Utility_Ty"] is not None:
        ut = row["Utility_Ty"]
        if ut in [1, 2, 11]:
            return None, {1:"Pole", 2:"Tower", 11:"Station"}[ut], "assets"

    return None, None, None


# -------------------------------
# PROCESS EACH LAYER
# -------------------------------
all_data = []

for layer in layers:
    try:
        gdf = gpd.read_file(input_path, layer=layer)

        # ❌ Skip empty layers
        if gdf.empty:
            print(f"Skipping empty layer: {layer}")
            continue

        # ❌ Skip layers with no useful attributes
        if len(gdf.columns) <= 1:
            print(f"Skipping no-attribute layer: {layer}")
            continue

        print(f"Processing layer: {layer}")

        # Apply classification
        gdf[["class_id", "subclass", "layer_name"]] = gdf.apply(
            lambda row: assign_classes(row), axis=1, result_type="expand"
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
    crs=all_data[0].crs
)

# -------------------------------
# SPLIT BY VILLAGE
# -------------------------------
villages = merged_gdf["Village_Na"].dropna().unique()

# for village in villages:
#     village_gdf = merged_gdf[merged_gdf["Village_Na"] == village]
#
#     output_path = os.path.join(output_dir, f"{village}.gpkg")
#
#     print(f"Saving village: {village}")
#
#     for layer_name in ["buildings", "roads", "water", "drainage", "assets"]:
#         layer_gdf = village_gdf[village_gdf["layer_name"] == layer_name]
#
#         if not layer_gdf.empty:
#             layer_gdf.to_file(
#                 output_path,
#                 layer=layer_name,
#                 driver="GPKG"
#             )

for village in villages:
    village_gdf = merged_gdf[merged_gdf["Village_Na"] == village]

    output_path = os.path.join(output_dir, f"{village}.gpkg")

    print(f"Saving village: {village}")

    # ---------------------------
    # GROUP BY layer_name FIRST
    # ---------------------------
    grouped = {
        "buildings": village_gdf[village_gdf["layer_name"] == "buildings"],
        "roads": village_gdf[village_gdf["layer_name"] == "roads"],
        "water": village_gdf[village_gdf["layer_name"] == "water"],
        "drainage": village_gdf[village_gdf["layer_name"] == "drainage"],
        "assets": village_gdf[village_gdf["layer_name"] == "assets"],
    }

    # ---------------------------
    # WRITE EACH LAYER ONLY ONCE
    # ---------------------------
    for layer_name, layer_gdf in grouped.items():
        if not layer_gdf.empty:
            layer_gdf.to_file(
                output_path,
                layer=layer_name,
                driver="GPKG"
            )

print("✅ Done!")