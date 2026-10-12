import unittest


try:
    from shapely import from_wkb, to_wkb
    from shapely.geometry import Point, Polygon

    from urban_workflow.silver.geospatial_operations import (
        NO_REPAIR_METHOD,
        REPAIR_METHOD,
        repair_district_boundary_records,
    )

    GEOSPATIAL_DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    GEOSPATIAL_DEPENDENCIES_AVAILABLE = False


def boundary_record(geometry: object) -> dict[str, object]:
    return {
        "COD_DIS": "1",
        "COD_DIS_TX": "01",
        "NOMBRE": "Centro",
        "DISTRI_MAY": "CENTRO",
        "DISTRI_MT": "CENTRO",
        "geometry_wkb": bytes(to_wkb(geometry, byte_order=1)),
        "geometry_type": geometry.geom_type,
        "geometry_crs": "EPSG:4326",
        "geometry_crs_source": "explicit_contract",
        "source_layer": "Distritos",
        "_source_id": "districts",
        "_source_sha256": "abc",
        "_bronze_run_id": "bronze-test",
    }


@unittest.skipUnless(
    GEOSPATIAL_DEPENDENCIES_AVAILABLE,
    "PyProj and Shapely are required for Silver geospatial tests.",
)
class SilverGeospatialOperationsTests(unittest.TestCase):
    def test_preserves_valid_polygon_without_repair(self) -> None:
        source = Polygon([(-3.8, 40.3), (-3.7, 40.3), (-3.7, 40.4), (-3.8, 40.4)])

        result = repair_district_boundary_records([boundary_record(source)])
        record = result.records[0]

        self.assertEqual(result.repaired_count, 0)
        self.assertTrue(record["source_geometry_was_valid"])
        self.assertFalse(record["geometry_repaired"])
        self.assertEqual(record["geometry_repair_method"], NO_REPAIR_METHOD)
        self.assertEqual(record["geometry_wkb"], record["source_geometry_wkb"])
        self.assertGreater(record["area_m2"], 0)

    def test_accepts_spark_binary_as_bytearray(self) -> None:
        source = Polygon([(-3.8, 40.3), (-3.7, 40.3), (-3.7, 40.4), (-3.8, 40.4)])
        record = boundary_record(source)
        record["geometry_wkb"] = bytearray(record["geometry_wkb"])

        result = repair_district_boundary_records([record])

        self.assertIsInstance(result.records[0]["source_geometry_wkb"], bytes)
        self.assertEqual(
            from_wkb(result.records[0]["source_geometry_wkb"]),
            source,
        )

    def test_repairs_self_intersection_to_valid_polygonal_geometry(self) -> None:
        source = Polygon([(-3.8, 40.3), (-3.7, 40.4), (-3.8, 40.4), (-3.7, 40.3)])

        result = repair_district_boundary_records([boundary_record(source)])
        record = result.records[0]
        repaired = from_wkb(record["geometry_wkb"])
        analysis = from_wkb(record["analysis_geometry_wkb"])

        self.assertEqual(result.source_invalid_count, 1)
        self.assertEqual(result.repaired_count, 1)
        self.assertFalse(record["source_geometry_was_valid"])
        self.assertTrue(record["geometry_repaired"])
        self.assertEqual(record["geometry_repair_method"], REPAIR_METHOD)
        self.assertTrue(repaired.is_valid)
        self.assertIn(repaired.geom_type, {"Polygon", "MultiPolygon"})
        self.assertTrue(analysis.is_valid)
        self.assertEqual(record["analysis_geometry_crs"], "EPSG:3035")

    def test_rejects_non_polygonal_source_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported source geometry"):
            repair_district_boundary_records([boundary_record(Point(-3.7, 40.4))])


if __name__ == "__main__":
    unittest.main()
