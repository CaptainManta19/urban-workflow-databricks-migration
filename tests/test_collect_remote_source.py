import json
import tempfile
import unittest
from pathlib import Path

from urban_workflow.ingestion.collect_remote_source import (
    DownloadedHTTPResponse,
    build_landing_path,
    build_request_url,
    collect_remote_source,
    inspect_ckan_page,
    validate_manually_uploaded_remote_source,
)
from urban_workflow.ingestion.schemas import SourceSpecification


def build_population_specification() -> SourceSpecification:
    return SourceSpecification(
        source_id="population_api",
        acquisition_mode="api",
        source_type="tabular",
        file_format="json",
        url="https://example.test/api/3/action/datastore_search",
        params={"resource_id": "population-resource"},
        expected_file_name="population.json",
    )


def build_ckan_payload(record_count: int = 5, total: int = 100) -> bytes:
    payload = {
        "success": True,
        "result": {
            "total": total,
            "limit": record_count,
            "offset": 0,
            "records": [{"id": index} for index in range(record_count)],
        },
    }
    return json.dumps(payload).encode("utf-8")


class RemoteSourceCollectionTests(unittest.TestCase):
    def test_builds_encoded_request_url(self) -> None:
        request_url = build_request_url(
            "https://example.test/api",
            {"resource_id": "Madrid population", "limit": "5"},
        )

        self.assertIn("resource_id=Madrid+population", request_url)
        self.assertIn("limit=5", request_url)

    def test_inspects_bounded_ckan_page(self) -> None:
        inspection = inspect_ckan_page(build_ckan_payload(record_count=5, total=1071))

        self.assertTrue(inspection.success)
        self.assertEqual(inspection.page_record_count, 5)
        self.assertEqual(inspection.total_record_count, 1071)
        self.assertEqual(inspection.offset, 0)

    def test_builds_source_and_run_specific_landing_path(self) -> None:
        path = build_landing_path(
            build_population_specification(),
            Path("/volume/raw_sources"),
            "run-123",
        )

        self.assertEqual(
            path,
            Path(
                "/volume/raw_sources/population_api/"
                "run_id=run-123/population.json"
            ),
        )

    def test_collects_page_without_overwriting_existing_artifact(self) -> None:
        downloaded_payload = build_ckan_payload(record_count=5, total=1071)

        def fake_fetcher(request_url: str, timeout_seconds: int) -> DownloadedHTTPResponse:
            self.assertIn("limit=5", request_url)
            self.assertEqual(timeout_seconds, 60)
            return DownloadedHTTPResponse(
                payload=downloaded_payload,
                status_code=200,
                content_type="application/json",
            )

        with tempfile.TemporaryDirectory() as directory:
            landing_root = Path(directory)
            specification = build_population_specification()
            result = collect_remote_source(
                specification=specification,
                run_id="run-123",
                landing_root=landing_root,
                request_parameter_overrides={"limit": "5"},
                fetcher=fake_fetcher,
            )

            self.assertEqual(result.event.status, "succeeded")
            self.assertEqual(result.event.record_count, 5)
            self.assertEqual(len(result.event.sha256 or ""), 64)
            self.assertEqual(result.ckan_page.total_record_count, 1071)
            self.assertEqual(len(result.event.warnings), 1)
            self.assertTrue(Path(result.event.landed_path or "").is_file())

            repeated_result = collect_remote_source(
                specification=specification,
                run_id="run-123",
                landing_root=landing_root,
                request_parameter_overrides={"limit": "5"},
                fetcher=fake_fetcher,
            )

            self.assertEqual(repeated_result.event.status, "failed")
            self.assertIn("already exists", repeated_result.event.error_message or "")

    def test_rejects_invalid_ckan_response_before_landing(self) -> None:
        def fake_fetcher(request_url: str, timeout_seconds: int) -> DownloadedHTTPResponse:
            return DownloadedHTTPResponse(
                payload=b'{"success": false}',
                status_code=200,
                content_type="application/json",
            )

        with tempfile.TemporaryDirectory() as directory:
            landing_root = Path(directory)
            result = collect_remote_source(
                specification=build_population_specification(),
                run_id="invalid-run",
                landing_root=landing_root,
                fetcher=fake_fetcher,
            )

            self.assertEqual(result.event.status, "failed")
            self.assertFalse(any(landing_root.rglob("*.json")))

    def test_validates_complete_manually_uploaded_ckan_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            landing_root = Path(directory)
            snapshot_path = landing_root / "population" / "population.json"
            snapshot_path.parent.mkdir(parents=True)
            snapshot_path.write_bytes(build_ckan_payload(record_count=5, total=5))
            specification = build_population_specification().model_copy(
                update={"manual_upload_path": str(snapshot_path)}
            )

            result = validate_manually_uploaded_remote_source(
                specification=specification,
                run_id="legacy-snapshot",
                landing_root=landing_root,
            )

            self.assertEqual(result.event.status, "succeeded")
            self.assertEqual(result.event.collection_method, "manual_upload")
            self.assertEqual(result.event.record_count, 5)
            self.assertIsNone(result.event.retrieved_at)
            self.assertIsNotNone(result.event.validated_at)
            self.assertEqual(result.ckan_page.page_record_count, 5)

    def test_rejects_incomplete_manually_uploaded_ckan_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            landing_root = Path(directory)
            snapshot_path = landing_root / "population.json"
            snapshot_path.write_bytes(build_ckan_payload(record_count=5, total=100))
            specification = build_population_specification().model_copy(
                update={"manual_upload_path": str(snapshot_path)}
            )

            result = validate_manually_uploaded_remote_source(
                specification=specification,
                run_id="legacy-snapshot",
                landing_root=landing_root,
            )

            self.assertEqual(result.event.status, "failed")
            self.assertIn("incomplete", result.event.error_message or "")

    def test_validates_manually_uploaded_topojson_snapshot(self) -> None:
        topojson_payload = {
            "type": "Topology",
            "objects": {
                "districts": {
                    "type": "GeometryCollection",
                    "geometries": [
                        {"type": "Polygon", "arcs": [[0]]},
                        {"type": "Polygon", "arcs": [[1]]},
                    ],
                }
            },
            "arcs": [[[0, 0], [1, 1]], [[1, 1], [2, 2]]],
        }

        with tempfile.TemporaryDirectory() as directory:
            landing_root = Path(directory)
            snapshot_path = landing_root / "boundaries.json"
            snapshot_path.write_text(json.dumps(topojson_payload), encoding="utf-8")
            specification = SourceSpecification(
                source_id="district_boundaries",
                acquisition_mode="remote_file",
                source_type="geospatial",
                file_format="json",
                url="https://example.test/boundaries.json",
                manual_upload_path=str(snapshot_path),
            )

            result = validate_manually_uploaded_remote_source(
                specification=specification,
                run_id="legacy-snapshot",
                landing_root=landing_root,
            )

            self.assertEqual(result.event.status, "succeeded")
            self.assertEqual(result.event.record_count, 2)
            self.assertEqual(result.json_file.root_type, "Topology")
            self.assertEqual(result.json_file.object_names, ["districts"])

    def test_reports_missing_manually_uploaded_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            landing_root = Path(directory)
            missing_path = landing_root / "missing.json"
            specification = build_population_specification().model_copy(
                update={"manual_upload_path": str(missing_path)}
            )

            result = validate_manually_uploaded_remote_source(
                specification=specification,
                run_id="legacy-snapshot",
                landing_root=landing_root,
            )

            self.assertEqual(result.event.status, "failed")
            self.assertIn("does not exist", result.event.error_message or "")


if __name__ == "__main__":
    unittest.main()
