"""Geospatial source adapters for portable Bronze WKB records."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyogrio
from pyproj import CRS


GEOMETRY_OUTPUT_COLUMNS = (
    "geometry_wkb",
    "geometry_type",
    "geometry_crs",
    "geometry_crs_source",
    "source_layer",
)


@dataclass(frozen=True)
class GeospatialConversion:
    """Driver-sized, Spark-ready records plus geometry conversion evidence."""

    records: list[dict[str, Any]]
    source_layer: str
    source_crs: str
    crs_source: str
    geometry_types: tuple[str, ...]
    invalid_geometry_count: int
    attribute_columns: tuple[str, ...]

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (*self.attribute_columns, *GEOMETRY_OUTPUT_COLUMNS)


def _available_layers(file_path: Path) -> list[str]:
    """Return layer names reported by GDAL through Pyogrio."""

    try:
        return [str(layer_name) for layer_name, _ in pyogrio.list_layers(file_path)]
    except Exception as exc:
        raise ValueError(f"Could not inspect geospatial source layers: {file_path}") from exc


def _read_layer(
    file_path: Path,
    *,
    layer: str,
    include_feature_id: bool,
) -> gpd.GeoDataFrame:
    """Read one explicitly selected layer and optionally preserve its feature ID."""

    if not file_path.is_file():
        raise FileNotFoundError(f"Geospatial source does not exist: {file_path}")

    available_layers = _available_layers(file_path)
    if layer not in available_layers:
        raise ValueError(
            f"Layer '{layer}' is not present in {file_path.name}. "
            f"Available layers: {available_layers}"
        )

    try:
        frame = gpd.read_file(
            file_path,
            layer=layer,
            engine="pyogrio",
            fid_as_index=include_feature_id,
        )
    except Exception as exc:
        raise ValueError(
            f"Could not decode layer '{layer}' from {file_path.name}."
        ) from exc

    if include_feature_id:
        feature_id_name = frame.index.name or "fid"
        if feature_id_name in frame.columns:
            raise ValueError(
                f"Feature ID name '{feature_id_name}' collides with a source attribute."
            )
        frame = frame.reset_index(names=feature_id_name)

    return frame


def _to_python_value(value: Any) -> Any:
    """Convert pandas and NumPy scalars into values Spark can infer reliably."""

    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _convert_to_wkb_records(
    frame: gpd.GeoDataFrame,
    *,
    source_layer: str,
    expected_crs: str,
    assign_crs_when_missing: bool,
    allowed_geometry_types: set[str],
    required_attribute_columns: set[str],
) -> GeospatialConversion:
    """Validate one GeoDataFrame and replace its active geometry with WKB."""

    if frame.empty:
        raise ValueError(f"Geospatial layer '{source_layer}' contains no records.")
    required_crs = CRS.from_user_input(expected_crs)
    if frame.crs is None:
        if not assign_crs_when_missing:
            raise ValueError(f"Geospatial layer '{source_layer}' does not declare a CRS.")
        if required_crs.to_epsg() != 4326:
            raise ValueError(
                "Automatic CRS assignment is limited to the explicit EPSG:4326 "
                "TopoJSON contract."
            )
        minimum_x, minimum_y, maximum_x, maximum_y = frame.total_bounds
        if not (
            -180 <= minimum_x <= maximum_x <= 180
            and -90 <= minimum_y <= maximum_y <= 90
        ):
            raise ValueError(
                f"Layer '{source_layer}' coordinates are outside longitude/latitude "
                "bounds and cannot be assigned EPSG:4326."
            )
        frame = frame.set_crs(required_crs)
        crs_source = "explicit_contract"
    else:
        crs_source = "source_metadata"

    actual_crs = CRS.from_user_input(frame.crs)
    if not actual_crs.equals(required_crs):
        raise ValueError(
            f"Unexpected CRS for layer '{source_layer}': "
            f"{actual_crs.to_string()}, expected {required_crs.to_string()}."
        )
    normalized_crs = required_crs.to_string()

    geometry_column = frame.geometry.name
    attribute_columns = [
        column_name for column_name in frame.columns if column_name != geometry_column
    ]
    missing_attributes = sorted(required_attribute_columns - set(attribute_columns))
    if missing_attributes:
        raise ValueError(
            f"Layer '{source_layer}' is missing required attributes: {missing_attributes}"
        )

    output_collisions = sorted(set(attribute_columns) & set(GEOMETRY_OUTPUT_COLUMNS))
    if output_collisions:
        raise ValueError(
            f"Source attributes collide with WKB output columns: {output_collisions}"
        )

    null_geometry_count = int(frame.geometry.isna().sum())
    if null_geometry_count:
        raise ValueError(
            f"Layer '{source_layer}' contains {null_geometry_count} null geometries."
        )

    geometry_types = tuple(sorted(set(frame.geometry.geom_type.astype(str))))
    unexpected_geometry_types = sorted(set(geometry_types) - allowed_geometry_types)
    if unexpected_geometry_types:
        raise ValueError(
            f"Layer '{source_layer}' contains unexpected geometry types: "
            f"{unexpected_geometry_types}; allowed: {sorted(allowed_geometry_types)}."
        )

    invalid_geometry_count = int((~frame.geometry.is_valid).sum())
    wkb_values = frame.geometry.to_wkb(hex=False, byte_order=1)
    if wkb_values.isna().any():
        raise ValueError(f"WKB conversion produced null values for layer '{source_layer}'.")

    records: list[dict[str, Any]] = []
    for row_position, (_, row) in enumerate(frame.iterrows()):
        geometry = frame.geometry.iloc[row_position]
        record = {
            column_name: _to_python_value(row[column_name])
            for column_name in attribute_columns
        }
        record.update(
            {
                "geometry_wkb": bytes(wkb_values.iloc[row_position]),
                "geometry_type": str(geometry.geom_type),
                "geometry_crs": normalized_crs,
                "geometry_crs_source": crs_source,
                "source_layer": source_layer,
            }
        )
        records.append(record)

    return GeospatialConversion(
        records=records,
        source_layer=source_layer,
        source_crs=normalized_crs,
        crs_source=crs_source,
        geometry_types=geometry_types,
        invalid_geometry_count=invalid_geometry_count,
        attribute_columns=tuple(attribute_columns),
    )


def read_topojson_as_wkb(
    file_path: Path,
    *,
    topology_object: str,
    expected_crs: str,
    allowed_geometry_types: set[str],
    required_attribute_columns: set[str],
) -> GeospatialConversion:
    """Decode one TopoJSON object and return portable WKB records."""

    frame = _read_layer(
        file_path,
        layer=topology_object,
        include_feature_id=False,
    )
    return _convert_to_wkb_records(
        frame,
        source_layer=topology_object,
        expected_crs=expected_crs,
        assign_crs_when_missing=True,
        allowed_geometry_types=allowed_geometry_types,
        required_attribute_columns=required_attribute_columns,
    )


def read_geopackage_as_wkb(
    file_path: Path,
    *,
    layer: str,
    expected_crs: str,
    allowed_geometry_types: set[str],
    required_attribute_columns: set[str],
) -> GeospatialConversion:
    """Read one GeoPackage feature layer and preserve its feature IDs in WKB records."""

    frame = _read_layer(
        file_path,
        layer=layer,
        include_feature_id=True,
    )
    return _convert_to_wkb_records(
        frame,
        source_layer=layer,
        expected_crs=expected_crs,
        assign_crs_when_missing=False,
        allowed_geometry_types=allowed_geometry_types,
        required_attribute_columns=required_attribute_columns,
    )
