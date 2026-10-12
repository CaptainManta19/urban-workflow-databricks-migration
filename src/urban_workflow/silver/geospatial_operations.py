"""Auditable geometry repair for the driver-sized Silver district layer."""

from dataclasses import dataclass
from typing import Any

from pyproj import CRS, Transformer
from shapely import from_wkb, make_valid, to_wkb, union_all
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.validation import explain_validity


SOURCE_CRS = CRS.from_epsg(4326)
ANALYSIS_CRS = CRS.from_epsg(3035)
REPAIR_METHOD = "make_valid_polygonal_components"
NO_REPAIR_METHOD = "not_required"


@dataclass(frozen=True)
class DistrictBoundaryConversion:
    """Spark-ready district records plus geometry-repair evidence."""

    records: list[dict[str, Any]]
    repaired_count: int
    source_invalid_count: int
    output_geometry_types: tuple[str, ...]
    source_crs: str
    analysis_crs: str

    @property
    def record_count(self) -> int:
        return len(self.records)


def _polygonal_components(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [
            polygon
            for part in geometry.geoms
            for polygon in _polygonal_components(part)
        ]
    return []


def _repair_polygonal_geometry(geometry: BaseGeometry) -> BaseGeometry:
    repaired = make_valid(geometry)
    polygonal_components = _polygonal_components(repaired)
    if not polygonal_components:
        raise ValueError("Geometry repair produced no polygonal components.")
    return union_all(polygonal_components)


def repair_district_boundary_records(
    source_records: list[dict[str, Any]],
) -> DistrictBoundaryConversion:
    """Repair invalid district polygons and create EPSG:4326/3035 WKB outputs."""

    if not source_records:
        raise ValueError("District boundary input contains no records.")

    required_fields = {
        "COD_DIS",
        "COD_DIS_TX",
        "NOMBRE",
        "DISTRI_MAY",
        "DISTRI_MT",
        "geometry_wkb",
        "geometry_type",
        "geometry_crs",
        "geometry_crs_source",
        "source_layer",
    }
    transformer = Transformer.from_crs(SOURCE_CRS, ANALYSIS_CRS, always_xy=True)
    records: list[dict[str, Any]] = []
    repaired_count = 0
    source_invalid_count = 0

    for source_record in source_records:
        missing_fields = sorted(required_fields - set(source_record))
        if missing_fields:
            raise ValueError(
                f"District boundary record is missing fields: {missing_fields}"
            )
        if source_record["geometry_crs"] != SOURCE_CRS.to_string():
            raise ValueError(
                "District boundary repair requires EPSG:4326 source WKB; found "
                f"{source_record['geometry_crs']!r}."
            )

        source_wkb = bytes(source_record["geometry_wkb"])
        source_geometry = from_wkb(source_wkb)
        if source_geometry.is_empty:
            raise ValueError(
                f"District {source_record['COD_DIS_TX']} has an empty source geometry."
            )
        if source_geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                f"District {source_record['COD_DIS_TX']} has unsupported source geometry "
                f"type {source_geometry.geom_type}."
            )

        source_was_valid = bool(source_geometry.is_valid)
        validity_reason = None if source_was_valid else explain_validity(source_geometry)
        if source_was_valid:
            clean_geometry = source_geometry
            repair_method = NO_REPAIR_METHOD
        else:
            source_invalid_count += 1
            clean_geometry = _repair_polygonal_geometry(source_geometry)
            repaired_count += 1
            repair_method = REPAIR_METHOD

        if (
            clean_geometry.is_empty
            or not clean_geometry.is_valid
            or clean_geometry.geom_type not in {"Polygon", "MultiPolygon"}
        ):
            raise ValueError(
                f"District {source_record['COD_DIS_TX']} did not produce a valid "
                "polygonal repair."
            )

        source_analysis_geometry = transform(transformer.transform, source_geometry)
        analysis_geometry = transform(transformer.transform, clean_geometry)
        if (
            analysis_geometry.is_empty
            or not analysis_geometry.is_valid
            or analysis_geometry.geom_type not in {"Polygon", "MultiPolygon"}
        ):
            raise ValueError(
                f"District {source_record['COD_DIS_TX']} is invalid after EPSG:3035 "
                "projection."
            )

        source_area_m2 = float(source_analysis_geometry.area)
        area_m2 = float(analysis_geometry.area)
        relative_area_change = (
            (area_m2 - source_area_m2) / source_area_m2
            if source_area_m2
            else None
        )
        records.append(
            {
                "district_code": source_record["COD_DIS_TX"],
                "district_number": int(source_record["COD_DIS"]),
                "district_name": source_record["NOMBRE"],
                "district_name_upper_source": source_record["DISTRI_MAY"],
                "district_name_display_source": source_record["DISTRI_MT"],
                "source_geometry_wkb": source_wkb,
                "source_geometry_type": source_geometry.geom_type,
                "source_geometry_crs": SOURCE_CRS.to_string(),
                "source_geometry_crs_source": source_record["geometry_crs_source"],
                "source_geometry_was_valid": source_was_valid,
                "source_geometry_validity_reason": validity_reason,
                "geometry_repaired": not source_was_valid,
                "geometry_repair_method": repair_method,
                "geometry_wkb": bytes(to_wkb(clean_geometry, byte_order=1)),
                "geometry_type": clean_geometry.geom_type,
                "geometry_crs": SOURCE_CRS.to_string(),
                "geometry_is_valid": bool(clean_geometry.is_valid),
                "analysis_geometry_wkb": bytes(
                    to_wkb(analysis_geometry, byte_order=1)
                ),
                "analysis_geometry_type": analysis_geometry.geom_type,
                "analysis_geometry_crs": ANALYSIS_CRS.to_string(),
                "analysis_geometry_is_valid": bool(analysis_geometry.is_valid),
                "source_area_m2": source_area_m2,
                "area_m2": area_m2,
                "repair_area_relative_change": relative_area_change,
                "source_layer": source_record["source_layer"],
                **{
                    key: value
                    for key, value in source_record.items()
                    if key.startswith("_")
                },
            }
        )

    output_geometry_types = tuple(
        sorted({record["geometry_type"] for record in records})
    )
    return DistrictBoundaryConversion(
        records=records,
        repaired_count=repaired_count,
        source_invalid_count=source_invalid_count,
        output_geometry_types=output_geometry_types,
        source_crs=SOURCE_CRS.to_string(),
        analysis_crs=ANALYSIS_CRS.to_string(),
    )
