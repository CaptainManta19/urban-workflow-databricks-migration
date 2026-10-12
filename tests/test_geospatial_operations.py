import json
import tempfile
import unittest
from pathlib import Path


try:
    import geopandas as gpd
    from shapely import from_wkb
    from shapely.geometry import Point, Polygon

    from urban_workflow.bronze.geospatial_operations import (
        read_geopackage_as_wkb,
        read_topojson_as_wkb,
    )

    GEOSPATIAL_DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    GEOSPATIAL_DEPENDENCIES_AVAILABLE = False


def write_minimal_topojson(file_path: Path) -> None:
    payload = {
        "type": "Topology",
        "transform": {"scale": [1.0, 1.0], "translate": [0.0, 0.0]},
        "objects": {
            "Distritos": {
                "type": "GeometryCollection",
                "geometries": [
                    {
                        "type": "Polygon",
                        "arcs": [[0]],
                        "properties": {
                            "COD_DIS": "1",
                            "COD_DIS_TX": "01",
                            "NOMBRE": "Centro",
                        },
                    }
                ],
            }
        },
        "arcs": [[[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]],
    }
    file_path.write_text(json.dumps(payload), encoding="utf-8")


@unittest.skipUnless(
    GEOSPATIAL_DEPENDENCIES_AVAILABLE,
    "GeoPandas, Pyogrio, PyProj, and Shapely are required for geospatial tests.",
)
class GeospatialOperationsTests(unittest.TestCase):
    def test_decodes_topojson_and_records_explicit_crs_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "districts.json"
            write_minimal_topojson(source_path)

            result = read_topojson_as_wkb(
                source_path,
                topology_object="Distritos",
                expected_crs="EPSG:4326",
                allowed_geometry_types={"Polygon", "MultiPolygon"},
                required_attribute_columns={"COD_DIS", "COD_DIS_TX", "NOMBRE"},
            )

            self.assertEqual(result.record_count, 1)
            self.assertEqual(result.source_crs, "EPSG:4326")
            self.assertEqual(result.crs_source, "explicit_contract")
            self.assertEqual(result.geometry_types, ("Polygon",))
            self.assertEqual(result.records[0]["NOMBRE"], "Centro")
            self.assertEqual(from_wkb(result.records[0]["geometry_wkb"]).geom_type, "Polygon")

    def test_reads_geopackage_and_preserves_feature_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "grid.gpkg"
            source_frame = gpd.GeoDataFrame(
                {
                    "cell_id": [0, 1],
                    "district_name": ["Centro", "Retiro"],
                },
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                ],
                crs="EPSG:3035",
            )
            source_frame.to_file(source_path, layer="grid_cells", driver="GPKG")

            result = read_geopackage_as_wkb(
                source_path,
                layer="grid_cells",
                expected_crs="EPSG:3035",
                allowed_geometry_types={"Polygon", "MultiPolygon"},
                required_attribute_columns={"fid", "cell_id", "district_name"},
            )

            self.assertEqual(result.record_count, 2)
            self.assertEqual(result.source_crs, "EPSG:3035")
            self.assertEqual(result.crs_source, "source_metadata")
            self.assertEqual([record["fid"] for record in result.records], [1, 2])
            self.assertEqual(
                [record["geometry_type"] for record in result.records],
                ["Polygon", "Polygon"],
            )

    def test_rejects_missing_topojson_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "districts.json"
            write_minimal_topojson(source_path)

            with self.assertRaisesRegex(ValueError, "Available layers"):
                read_topojson_as_wkb(
                    source_path,
                    topology_object="Barrios",
                    expected_crs="EPSG:4326",
                    allowed_geometry_types={"Polygon", "MultiPolygon"},
                    required_attribute_columns={"NOMBRE"},
                )

    def test_rejects_unexpected_geometry_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "points.gpkg"
            source_frame = gpd.GeoDataFrame(
                {"cell_id": [0]},
                geometry=[Point(0, 0)],
                crs="EPSG:3035",
            )
            source_frame.to_file(source_path, layer="grid_cells", driver="GPKG")

            with self.assertRaisesRegex(ValueError, "unexpected geometry types"):
                read_geopackage_as_wkb(
                    source_path,
                    layer="grid_cells",
                    expected_crs="EPSG:3035",
                    allowed_geometry_types={"Polygon", "MultiPolygon"},
                    required_attribute_columns={"fid", "cell_id"},
                )

    def test_rejects_wrong_crs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "wrong_crs.gpkg"
            source_frame = gpd.GeoDataFrame(
                {"cell_id": [0]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                crs="EPSG:4326",
            )
            source_frame.to_file(source_path, layer="grid_cells", driver="GPKG")

            with self.assertRaisesRegex(ValueError, "Unexpected CRS"):
                read_geopackage_as_wkb(
                    source_path,
                    layer="grid_cells",
                    expected_crs="EPSG:3035",
                    allowed_geometry_types={"Polygon", "MultiPolygon"},
                    required_attribute_columns={"fid", "cell_id"},
                )


if __name__ == "__main__":
    unittest.main()
