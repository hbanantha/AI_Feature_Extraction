"""
GIS Output Pipeline for Feature Extraction
==========================================
Export predictions as georeferenced Shapefiles and GeoPackage formats.
Supports multiple feature classes with full attribute preservation.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from datetime import datetime

import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon, box
from rasterio.features import shapes
import rasterio
from rasterio.crs import CRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Feature class definitions
FEATURE_CLASSES = {
    0: {"name": "background", "export": False, "type": "other"},
    1: {"name": "building_rcc", "export": True, "type": "building"},
    2: {"name": "building_tiled", "export": True, "type": "building"},
    3: {"name": "building_tin", "export": True, "type": "building"},
    4: {"name": "building_others", "export": True, "type": "building"},
    5: {"name": "road", "export": True, "type": "infrastructure"},
    6: {"name": "waterbody", "export": True, "type": "water"},
}


class GISExporter:
    """
    Export segmentation predictions to GIS-compatible formats.
    Supports Shapefile and GeoPackage outputs with full attribute preservation.
    """

    def __init__(
        self,
        output_dir: str,
        crs: Optional[CRS] = None,
        min_polygon_area: float = 10.0,
        min_line_length: float = 5.0,
        config: Optional[Dict] = None
    ):
        """
        Initialize GIS exporter.

        Args:
            output_dir: Output directory for GIS files
            crs: Coordinate Reference System (default: EPSG:4326 - WGS84)
            min_polygon_area: Minimum polygon area in square meters
            min_line_length: Minimum feature length in meters
            config: Configuration dictionary for class-specific settings
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default to WGS84 if not specified
        self.crs = crs or CRS.from_epsg(4326)
        
        self.min_polygon_area = min_polygon_area
        self.min_line_length = min_line_length
        self.config = config or {}

        # Store all GeoDataFrames for GeoPackage creation
        self.gdfs = {}
        
        logger.info(f"GIS Exporter initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"CRS: {self.crs}")

    def export_predictions(
        self,
        predictions: np.ndarray,
        transform: Any,
        output_name: str,
        confidence: Optional[np.ndarray] = None
    ) -> Dict[str, Path]:
        """
        Export predictions as both Shapefiles and GeoPackage.

        Args:
            predictions: Class prediction array (H, W)
            transform: Affine transform for georeferencing
            output_name: Base name for output files
            confidence: Confidence scores array (H, W) - optional

        Returns:
            Dictionary with output file paths
        """
        output_paths = {}
        
        logger.info(f"Exporting predictions for: {output_name}")
        logger.info(f"Prediction shape: {predictions.shape}")

        # Clean predictions
        predictions = self._clean_predictions(predictions)

        # Extract geometries for each class
        all_layers = {}
        
        for class_idx, class_info in FEATURE_CLASSES.items():
            if not class_info["export"]:
                continue

            class_name = class_info["name"]
            feature_type = class_info["type"]

            logger.info(f"Processing class: {class_name}")

            # Extract geometries
            gdf = self._extract_class_geometries(
                predictions, class_idx, class_name,
                transform, confidence
            )

            if gdf is not None and len(gdf) > 0:
                all_layers[class_name] = gdf
                
                # Save individual shapefile
                shp_path = self._save_shapefile(gdf, output_name, class_name)
                output_paths[f"shapefile_{class_name}"] = shp_path
                
                # Store for GeoPackage
                self.gdfs[class_name] = gdf

        # Create combined GeoPackage
        if all_layers:
            gpkg_path = self._save_geopackage(
                all_layers, output_name
            )
            output_paths["geopackage"] = gpkg_path
        else:
            logger.warning("No valid geometries extracted for any class")

        # Save export metadata
        meta_path = self._save_export_metadata(
            output_paths, predictions, output_name
        )
        output_paths["metadata"] = meta_path

        logger.info(f"Export complete. Files saved to: {self.output_dir}")
        return output_paths

    def _clean_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean predictions.

        Args:
            predictions: Class prediction array

        Returns:
            Cleaned predictions
        """
        cleaned = predictions.copy()

        # Apply morphological closing to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for class_idx in np.unique(predictions):
            if class_idx == 0:
                continue

            mask = (predictions == class_idx).astype(np.uint8)

            # -------------------------------
            #  SPECIAL HANDLING FOR ROADS
            # -------------------------------
            if class_idx == 5:  # road
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask = cv2.dilate(mask, kernel, iterations=1)

            # -------------------------------
            #  WATER SMOOTHING
            # -------------------------------
            elif class_idx == 6:  # waterbody
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # -------------------------------
            #  BUILDINGS CLEANING
            # -------------------------------
            else:  # buildings
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # Final smoothing
            mask = cv2.medianBlur(mask, 5)

            cleaned[mask > 0] = class_idx

        return cleaned

    def _extract_class_geometries(
        self,
        predictions: np.ndarray,
        class_idx: int,
        class_name: str,
        transform: Any,
        confidence: Optional[np.ndarray] = None
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Extract geometries for a specific class.

        Args:
            predictions: Class prediction array
            class_idx: Class index
            class_name: Class name
            transform: Affine transform
            confidence: Confidence scores

        Returns:
            GeoDataFrame with geometries
        """
        # Create binary mask
        mask = (predictions == class_idx).astype(np.uint8)

        # Extract geometries
        geometries = []
        properties = []

        try:
            for geom, value in shapes(mask, transform=transform):
                if value == 1:
                    polygon = shape(geom)

                    # -------------------------------
                    #  STEP 1: Fix invalid geometry
                    # -------------------------------
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)

                    if polygon.is_empty or polygon.area == 0:
                        continue

                    # -------------------------------
                    #  STEP 2: Remove tiny noise early
                    # -------------------------------
                    if polygon.area < (self.min_polygon_area * 0.3):
                        continue

                    # -------------------------------
                    #  STEP 3: Smooth edges
                    # -------------------------------
                    polygon = polygon.buffer(1)
                    polygon = polygon.simplify(1.5, preserve_topology=True)
                    polygon = polygon.buffer(-1)

                    # -------------------------------
                    #  STEP 4: Remove small holes
                    # -------------------------------
                    if polygon.geom_type == "Polygon":
                        polygon = Polygon(
                            polygon.exterior,
                            [hole for hole in polygon.interiors if Polygon(hole).area > self.min_polygon_area]
                        )

                    # -------------------------------
                    #  STEP 5: Final validity check
                    # -------------------------------
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)

                    if polygon.is_empty:
                        continue

                    # Apply area/length filters
                    if not self._passes_filter(polygon, class_name):
                        continue

                    geometries.append(polygon)
                    properties.append(
                        self._get_geometry_properties(
                            polygon, class_name, class_idx,
                            confidence, mask, transform
                        )
                    )

        except Exception as e:
            logger.error(f"Error extracting geometries for {class_name}: {e}")
            return None

        if not geometries:
            logger.warning(f"No valid geometries found for class: {class_name}")
            return None

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            properties,
            geometry=geometries,
            crs=self.crs
        )
        try:
            merged = gdf.unary_union

            if merged.geom_type == "MultiPolygon":
                geometries = list(merged.geoms)
            else:
                geometries = [merged]

            gdf = gpd.GeoDataFrame(
                geometry=geometries,
                crs=self.crs
            )

        except Exception as e:
            logger.warning(f"Merge failed: {e}")
        logger.info(f"Extracted {len(gdf)} geometries for {class_name}")
        return gdf

    def _passes_filter(self, geometry: Polygon, class_name: str) -> bool:
        """
        Check if geometry passes size filters.

        Args:
            geometry: Shapely geometry
            class_name: Class name

        Returns:
            True if geometry passes filters
        """
        if "building" in class_name:
            return geometry.area >= self.min_polygon_area
        elif class_name == "road":
            return geometry.length >= self.min_line_length
        elif class_name == "waterbody":
            return geometry.area >= (self.min_polygon_area * 0.5)
        else:
            return geometry.area > 0

    def _get_geometry_properties(
        self,
        geometry: Polygon,
        class_name: str,
        class_idx: int,
        confidence: Optional[np.ndarray],
        mask: np.ndarray,
        transform: Any
    ) -> Dict[str, Any]:
        """
        Get properties for a geometry.

        Args:
            geometry: Shapely geometry
            class_name: Class name
            class_idx: Class index
            confidence: Confidence array
            mask: Binary mask
            transform: Affine transform

        Returns:
            Dictionary of properties
        """
        properties = {
            "id": None,  # Will be set on export
            "class": class_name,
            "class_id": class_idx,
            "area_m2": geometry.area,
            "perimeter_m": geometry.length,
        }

        # Add confidence if available
        if confidence is not None:
            try:
                bounds = geometry.bounds
                # Get average confidence for this geometry
                properties["avg_confidence"] = float(
                    np.mean(confidence[mask > 0]) if np.any(mask > 0) else 0
                )
            except Exception:
                properties["avg_confidence"] = 0.0

        # Add geometry-specific properties
        if hasattr(geometry, "exterior"):
            properties["num_vertices"] = len(geometry.exterior.coords)

        return properties

    def _save_shapefile(
        self,
        gdf: gpd.GeoDataFrame,
        output_name: str,
        class_name: str
    ) -> Path:
        """
        Save GeoDataFrame as Shapefile.

        Args:
            gdf: GeoDataFrame to save
            output_name: Base output name
            class_name: Class name

        Returns:
            Path to saved shapefile
        """
        # Create unique IDs
        gdf = gdf.copy()
        gdf["id"] = range(1, len(gdf) + 1)

        # Save shapefile
        output_path = self.output_dir / f"{output_name}_{class_name}.shp"
        
        try:
            gdf.to_file(output_path, driver="ESRI Shapefile")
            logger.info(f"Saved shapefile: {output_path} ({len(gdf)} features)")
        except Exception as e:
            logger.error(f"Error saving shapefile {output_path}: {e}")
            return None

        return output_path

    def _save_geopackage(
        self,
        layers: Dict[str, gpd.GeoDataFrame],
        output_name: str
    ) -> Path:
        """
        Save all layers to a single GeoPackage file.

        Args:
            layers: Dictionary of layer_name -> GeoDataFrame
            output_name: Base output name

        Returns:
            Path to saved GeoPackage
        """
        output_path = self.output_dir / f"{output_name}_features.gpkg"

        try:
            for layer_name, gdf in layers.items():
                # Create unique IDs
                gdf_copy = gdf.copy()
                gdf_copy["id"] = range(1, len(gdf_copy) + 1)

                # Ensure CRS is set
                if gdf_copy.crs is None:
                    gdf_copy = gdf_copy.set_crs(self.crs)

                # Write to GeoPackage
                if not output_path.exists():
                    # First layer - create new
                    gdf_copy.to_file(
                        output_path,
                        layer=layer_name,
                        driver="GPKG"
                    )
                else:
                    # Append subsequent layers
                    gdf_copy.to_file(
                        output_path,
                        layer=layer_name,
                        driver="GPKG"
                    )

                logger.info(f"Added layer '{layer_name}' to GeoPackage: {len(gdf_copy)} features")

        except Exception as e:
            logger.error(f"Error saving GeoPackage {output_path}: {e}")
            return None

        logger.info(f"GeoPackage saved: {output_path} ({len(layers)} layers)")
        return output_path

    def _save_export_metadata(
        self,
        output_paths: Dict[str, Path],
        predictions: np.ndarray,
        output_name: str
    ) -> Path:
        """
        Save export metadata.

        Args:
            output_paths: Dictionary of output file paths
            predictions: Prediction array
            output_name: Output name

        Returns:
            Path to metadata file
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "output_name": output_name,
            "prediction_shape": predictions.shape,
            "crs": str(self.crs),
            "output_files": {
                k: str(v) for k, v in output_paths.items()
            },
            "class_statistics": self._get_class_statistics(predictions),
            "export_settings": {
                "min_polygon_area": self.min_polygon_area,
                "min_line_length": self.min_line_length,
            }
        }

        meta_path = self.output_dir / f"{output_name}_export_metadata.json"
        
        try:
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Export metadata saved: {meta_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return None

        return meta_path

    def _get_class_statistics(self, predictions: np.ndarray) -> Dict[str, int]:
        """Get pixel count statistics for each class."""
        stats = {}
        for class_idx, class_info in FEATURE_CLASSES.items():
            count = np.sum(predictions == class_idx)
            if count > 0:
                stats[class_info["name"]] = int(count)
        return stats

    def create_merged_geopackage(
        self,
        gpkg_paths: List[Path],
        output_name: str
    ) -> Path:
        """
        Merge multiple GeoPackage files into a single file.

        Args:
            gpkg_paths: List of GeoPackage file paths
            output_name: Name for merged output

        Returns:
            Path to merged GeoPackage
        """
        output_path = self.output_dir / f"{output_name}_merged_features.gpkg"
        
        all_layers = {}
        layer_counter = {}

        try:
            for gpkg_path in gpkg_paths:
                # Read all layers from GeoPackage
                gdf_all = gpd.read_file(gpkg_path)
                
                # Get layer names
                import fiona
                layer_names = fiona.listlayers(str(gpkg_path))
                
                for layer_name in layer_names:
                    gdf = gpd.read_file(gpkg_path, layer=layer_name)
                    
                    # Handle duplicate layer names
                    if layer_name in all_layers:
                        layer_counter[layer_name] = layer_counter.get(layer_name, 1) + 1
                        unique_name = f"{layer_name}_{layer_counter[layer_name]}"
                    else:
                        unique_name = layer_name
                        layer_counter[layer_name] = 0
                    
                    all_layers[unique_name] = gdf

            # Write merged GeoPackage
            for idx, (layer_name, gdf) in enumerate(all_layers.items()):
                if idx == 0:
                    gdf.to_file(output_path, layer=layer_name, driver="GPKG")
                else:
                    gdf.to_file(output_path, layer=layer_name, driver="GPKG")

            logger.info(f"Merged GeoPackage saved: {output_path} ({len(all_layers)} layers)")
            return output_path

        except Exception as e:
            logger.error(f"Error creating merged GeoPackage: {e}")
            return None

    def export_to_geojson(
        self,
        gdf: gpd.GeoDataFrame,
        output_name: str,
        class_name: str
    ) -> Path:
        """
        Export GeoDataFrame as GeoJSON.

        Args:
            gdf: GeoDataFrame to export
            output_name: Base output name
            class_name: Class name

        Returns:
            Path to saved GeoJSON
        """
        output_path = self.output_dir / f"{output_name}_{class_name}.geojson"
        
        try:
            gdf.to_file(output_path, driver="GeoJSON")
            logger.info(f"GeoJSON saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving GeoJSON: {e}")
            return None

    def get_layer_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary of all layers.

        Returns:
            Dictionary with layer information
        """
        summary = {}
        for layer_name, gdf in self.gdfs.items():
            summary[layer_name] = {
                "feature_count": len(gdf),
                "crs": str(gdf.crs),
                "bounds": gdf.total_bounds.tolist(),
                "attributes": list(gdf.columns),
            }
        return summary

    def validate_exports(self) -> Dict[str, bool]:
        """
        Validate exported files.

        Returns:
            Dictionary with validation results
        """
        results = {}
        
        for file_path in self.output_dir.glob("*.shp"):
            try:
                gdf = gpd.read_file(file_path)
                results[file_path.name] = len(gdf) > 0 and gdf.crs is not None
            except Exception as e:
                logger.warning(f"Validation failed for {file_path}: {e}")
                results[file_path.name] = False

        for file_path in self.output_dir.glob("*.gpkg"):
            try:
                import fiona
                layers = fiona.listlayers(str(file_path))
                results[file_path.name] = len(layers) > 0
            except Exception as e:
                logger.warning(f"Validation failed for {file_path}: {e}")
                results[file_path.name] = False

        return results

