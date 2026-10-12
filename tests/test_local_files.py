import sqlite3
import tempfile
import unittest
from pathlib import Path

from urban_workflow.ingestion.collect_local_source import (
    collect_local_source,
    inspect_csv,
    inspect_geopackage,
)
from urban_workflow.ingestion.provenance import ensure_file_is_materialised
from urban_workflow.ingestion.schemas import (
    CSVReadOptions,
    SourceManifest,
    SourceSpecification,
    resolve_manifest_paths,
)


def create_minimal_geopackage(file_path: Path) -> None:
    with sqlite3.connect(file_path) as connection:
        connection.executescript(
            """
            CREATE TABLE gpkg_contents (
                table_name TEXT PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT,
                description TEXT DEFAULT '',
                last_change DATETIME,
                min_x DOUBLE,
                min_y DOUBLE,
                max_x DOUBLE,
                max_y DOUBLE,
                srs_id INTEGER
            );
            CREATE TABLE gpkg_geometry_columns (
                table_name TEXT PRIMARY KEY,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL
            );
            CREATE TABLE gpkg_spatial_ref_sys (
                srs_name TEXT NOT NULL,
                srs_id INTEGER PRIMARY KEY,
                organization TEXT NOT NULL,
                organization_coordsys_id INTEGER NOT NULL,
                definition TEXT NOT NULL,
                description TEXT
            );
            CREATE TABLE grid_cells (
                fid INTEGER PRIMARY KEY,
                geom BLOB,
                cell_id TEXT,
                district_name TEXT
            );
            INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id)
            VALUES ('grid_cells', 'features', 'grid_cells', 3035);
            INSERT INTO gpkg_geometry_columns
                (table_name, column_name, geometry_type_name, srs_id, z, m)
            VALUES ('grid_cells', 'geom', 'POLYGON', 3035, 0, 0);
            INSERT INTO gpkg_spatial_ref_sys
                (srs_name, srs_id, organization, organization_coordsys_id, definition)
            VALUES ('ETRS89-extended / LAEA Europe', 3035, 'EPSG', 3035, 'test');
            INSERT INTO grid_cells (cell_id, district_name) VALUES ('1', 'Centro');
            INSERT INTO grid_cells (cell_id, district_name) VALUES ('2', 'Retiro');
            """
        )


class LocalFileCollectionTests(unittest.TestCase):
    def test_resolves_manifest_relative_paths_inside_landing_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            landing_root = Path(directory) / "raw_sources"
            manifest = SourceManifest(
                sources=[
                    SourceSpecification(
                        source_id="relative_source",
                        acquisition_mode="local_file",
                        source_type="tabular",
                        file_format="csv",
                        path="madrid/source.csv",
                    )
                ]
            )

            resolved = resolve_manifest_paths(manifest, landing_root)

            self.assertEqual(
                resolved.sources[0].path,
                str((landing_root / "madrid/source.csv").resolve()),
            )

    def test_rejects_manifest_relative_path_outside_landing_root(self) -> None:
        manifest = SourceManifest(
            sources=[
                SourceSpecification(
                    source_id="escaping_source",
                    acquisition_mode="local_file",
                    source_type="tabular",
                    file_format="csv",
                    path="../source.csv",
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "escapes the landing root"):
            resolve_manifest_paths(manifest, "/Volumes/catalog/schema/raw_sources")

    def test_detects_git_lfs_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pointer_path = Path(directory) / "source.gpkg"
            pointer_path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:0123456789abcdef\n"
                "size 3506176\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unresolved Git LFS pointer"):
                ensure_file_is_materialised(pointer_path)

    def test_inspects_geopackage_metadata_without_loading_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            geopackage_path = Path(directory) / "grid.gpkg"
            create_minimal_geopackage(geopackage_path)

            inspection = inspect_geopackage(geopackage_path)

            self.assertEqual(inspection.selected_layer, "grid_cells")
            self.assertEqual(inspection.row_count, 2)
            self.assertEqual(inspection.crs, "EPSG:3035")
            self.assertEqual(inspection.geometry_column, "geom")
            self.assertEqual(inspection.geometry_types, ["POLYGON"])

    def test_builds_successful_collection_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            landing_root = Path(directory)
            geopackage_path = landing_root / "grid.gpkg"
            create_minimal_geopackage(geopackage_path)
            specification = SourceSpecification(
                source_id="test_grid",
                acquisition_mode="local_file",
                source_kind="derived_snapshot",
                source_type="geospatial",
                file_format="geopackage",
                path=str(geopackage_path),
                upstream_sources=["test source"],
                transformation_reference="test notebook",
            )

            result = collect_local_source(
                specification=specification,
                run_id="test-run",
                landing_root=landing_root,
            )

            self.assertEqual(result.event.status, "succeeded")
            self.assertEqual(result.event.record_count, 2)
            self.assertEqual(len(result.event.sha256 or ""), 64)
            self.assertIsNotNone(result.provenance)
            assert result.provenance is not None
            self.assertEqual(result.provenance.relative_path, "grid.gpkg")

    def test_inspects_semicolon_csv_with_preamble_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "housing.csv"
            csv_path.write_text(
                "Housing report;;;;;\n"
                "DISTRICT;TOTAL;PROGRAM;;;;\n"
                "CENTRO;10;7;;;;\n"
                "RETIRO;20;15;;;;\n"
                "TOTAL;30;22;;;;\n",
                encoding="utf-8-sig",
            )

            inspection = inspect_csv(
                csv_path,
                CSVReadOptions(
                    encoding="utf-8-sig",
                    delimiter=";",
                    header_row=1,
                ),
            )

            self.assertEqual(inspection.row_count, 3)
            self.assertEqual(inspection.columns[:3], ["DISTRICT", "TOTAL", "PROGRAM"])
            self.assertEqual(inspection.column_types["TOTAL"], "string")
            self.assertEqual(inspection.empty_column_name_count, 4)

    def test_dispatches_csv_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            landing_root = Path(directory)
            csv_path = landing_root / "housing.csv"
            csv_path.write_text(
                "Report;;\nDISTRICT;TOTAL;\nCENTRO;10;\nTOTAL;10;\n",
                encoding="utf-8-sig",
            )
            specification = SourceSpecification(
                source_id="test_housing",
                acquisition_mode="local_file",
                source_type="tabular",
                file_format="csv",
                path=str(csv_path),
                csv_options=CSVReadOptions(
                    encoding="utf-8-sig",
                    delimiter=";",
                    header_row=1,
                ),
            )

            result = collect_local_source(
                specification=specification,
                run_id="test-run",
                landing_root=landing_root,
            )

            self.assertEqual(result.event.status, "succeeded")
            self.assertEqual(result.event.content_type, "text/csv")
            self.assertEqual(result.event.record_count, 2)
            self.assertIsNotNone(result.csv)
            self.assertIsNone(result.geopackage)

    def test_reports_unsupported_local_format(self) -> None:
        specification = SourceSpecification(
            source_id="test_json",
            acquisition_mode="local_file",
            source_type="tabular",
            file_format="json",
            path="not-used.json",
        )

        result = collect_local_source(
            specification=specification,
            run_id="test-run",
            landing_root=Path("."),
        )

        self.assertEqual(result.event.status, "failed")
        self.assertIn("does not yet support", result.event.error_message or "")


if __name__ == "__main__":
    unittest.main()
