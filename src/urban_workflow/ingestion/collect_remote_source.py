import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from urban_workflow.ingestion.provenance import (
    build_provenance,
    ensure_file_is_materialised,
)
from urban_workflow.ingestion.schemas import (
    CKANPageInspection,
    CollectionEvent,
    HTTPResponseInspection,
    JSONFileInspection,
    RemoteSourceCollectionResult,
    SourceSpecification,
)


DEFAULT_HTTP_TIMEOUT_SECONDS = 60
USER_AGENT = "UrbanWorkflowDatabricksMigration/0.1"


@dataclass(frozen=True)
class DownloadedHTTPResponse:
    payload: bytes
    status_code: int
    content_type: str | None


RemoteFetcher = Callable[[str, int], DownloadedHTTPResponse]


def build_request_url(base_url: str, parameters: dict[str, str]) -> str:
    """Build a request URL without modifying the manifest parameters."""

    if not parameters:
        return base_url

    query_string = urlencode(parameters)
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}{query_string}"


def fetch_http_response(
    request_url: str,
    timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> DownloadedHTTPResponse:
    """Fetch one HTTP response using only the Python standard library."""

    request = Request(request_url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout_seconds) as response:
        return DownloadedHTTPResponse(
            payload=response.read(),
            status_code=int(response.status),
            content_type=response.headers.get("Content-Type"),
        )


def inspect_ckan_page(payload: bytes) -> CKANPageInspection:
    """Validate the CKAN response envelope and summarize the current page."""

    try:
        parsed_payload = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Remote API response is not valid UTF-8 JSON.") from exc

    if not isinstance(parsed_payload, dict):
        raise ValueError("CKAN response must be a JSON object.")
    if parsed_payload.get("success") is not True:
        raise ValueError("CKAN response did not report success=true.")

    result = parsed_payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("CKAN response is missing the result object.")

    records = result.get("records")
    total = result.get("total")
    if not isinstance(records, list):
        raise ValueError("CKAN result.records must be a list.")
    if not isinstance(total, int):
        raise ValueError("CKAN result.total must be an integer.")

    limit = result.get("limit", len(records))
    offset = result.get("offset", 0)
    if not isinstance(limit, int) or not isinstance(offset, int):
        raise ValueError("CKAN result limit and offset values must be integers.")

    return CKANPageInspection(
        success=True,
        total_record_count=total,
        page_record_count=len(records),
        limit=limit,
        offset=offset,
    )


def inspect_json_file(file_path: Path, payload: bytes) -> JSONFileInspection:
    """Validate a JSON object and summarize TopoJSON structure when present."""

    try:
        parsed_payload = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Uploaded snapshot is not valid UTF-8 JSON.") from exc

    if not isinstance(parsed_payload, dict):
        raise ValueError("Uploaded JSON snapshot must contain a top-level object.")

    root_type = parsed_payload.get("type")
    if root_type is not None and not isinstance(root_type, str):
        raise ValueError("JSON field 'type' must be a string when present.")

    object_names: list[str] = []
    record_count: int | None = None
    if root_type == "Topology":
        topology_objects = parsed_payload.get("objects")
        topology_arcs = parsed_payload.get("arcs")
        if not isinstance(topology_objects, dict) or not topology_objects:
            raise ValueError("TopoJSON snapshot requires a non-empty 'objects' object.")
        if not isinstance(topology_arcs, list):
            raise ValueError("TopoJSON snapshot requires an 'arcs' list.")

        object_names = sorted(str(name) for name in topology_objects)
        record_count = 0
        for topology_object in topology_objects.values():
            if not isinstance(topology_object, dict):
                raise ValueError("Every TopoJSON object must be a JSON object.")
            geometries = topology_object.get("geometries")
            if isinstance(geometries, list):
                record_count += len(geometries)
            else:
                record_count += 1

    return JSONFileInspection(
        file_path=str(file_path.resolve()),
        top_level_keys=sorted(str(key) for key in parsed_payload),
        root_type=root_type,
        object_names=object_names,
        record_count=record_count,
    )


def validate_manually_uploaded_remote_source(
    specification: SourceSpecification,
    run_id: str,
    landing_root: Path,
    validated_at: datetime | None = None,
) -> RemoteSourceCollectionResult:
    """Validate a remote-origin snapshot that was manually uploaded to a volume."""

    event_time = validated_at or datetime.now(timezone.utc)
    snapshot_path = (
        Path(specification.manual_upload_path)
        if specification.manual_upload_path
        else None
    )

    try:
        if specification.acquisition_mode not in {"api", "remote_file"}:
            raise ValueError(
                "Manual remote validation requires acquisition_mode='api' or "
                "acquisition_mode='remote_file'."
            )
        if specification.file_format != "json":
            raise ValueError("Manual remote validation currently supports JSON files.")
        if snapshot_path is None:
            raise ValueError("Manual remote validation requires manual_upload_path.")
        if not snapshot_path.is_file():
            raise FileNotFoundError(
                f"Manually uploaded snapshot does not exist: {snapshot_path}"
            )

        resolved_landing_root = landing_root.resolve()
        resolved_snapshot_path = snapshot_path.resolve()
        try:
            resolved_snapshot_path.relative_to(resolved_landing_root)
        except ValueError as exc:
            raise ValueError(
                "manual_upload_path must be located inside the configured landing root."
            ) from exc

        ensure_file_is_materialised(snapshot_path)
        payload = snapshot_path.read_bytes()
        json_inspection = inspect_json_file(snapshot_path, payload)
        ckan_page = None

        if specification.acquisition_mode == "api":
            ckan_page = inspect_ckan_page(payload)
            if ckan_page.page_record_count != ckan_page.total_record_count:
                raise ValueError(
                    "Uploaded CKAN snapshot is incomplete: "
                    f"it contains {ckan_page.page_record_count} records but reports "
                    f"{ckan_page.total_record_count} total records."
                )
            record_count = ckan_page.page_record_count
        else:
            if json_inspection.root_type != "Topology":
                raise ValueError(
                    "The district boundary snapshot must be valid TopoJSON with "
                    "type='Topology'."
                )
            record_count = json_inspection.record_count

        provenance = build_provenance(snapshot_path, landing_root)
        event = CollectionEvent(
            run_id=run_id,
            source_id=specification.source_id,
            acquisition_mode=specification.acquisition_mode,
            collection_method="manual_upload",
            source_kind=specification.source_kind,
            origin_url=specification.url,
            landed_path=str(resolved_snapshot_path),
            validated_at=event_time,
            content_type="application/json",
            file_size_bytes=provenance.file_size_bytes,
            sha256=provenance.sha256,
            record_count=record_count,
            status="succeeded",
            warnings=[
                "This remote-origin snapshot was fetched by the legacy workflow and "
                "manually uploaded because Databricks Free Edition blocks its source "
                "domain. Its original retrieval time cannot be reconstructed from the "
                "uploaded file."
            ],
        )
        return RemoteSourceCollectionResult(
            event=event,
            provenance=provenance,
            ckan_page=ckan_page,
            json_file=json_inspection,
        )
    except Exception as exc:
        event = CollectionEvent(
            run_id=run_id,
            source_id=specification.source_id,
            acquisition_mode=specification.acquisition_mode,
            collection_method="manual_upload",
            source_kind=specification.source_kind,
            origin_url=specification.url,
            landed_path=str(snapshot_path) if snapshot_path else None,
            validated_at=event_time,
            status="failed",
            error_message=str(exc),
            warnings=[
                "The manually uploaded remote snapshot was not accepted. Review "
                "error_message before continuing."
            ],
        )
        return RemoteSourceCollectionResult(event=event)


def build_landing_path(
    specification: SourceSpecification,
    landing_root: Path,
    run_id: str,
) -> Path:
    """Build an immutable, source-specific landing path for one run."""

    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty path-safe value.")

    expected_file_name = specification.expected_file_name or f"{specification.source_id}.json"
    if Path(expected_file_name).name != expected_file_name:
        raise ValueError("expected_file_name must not contain a directory path.")

    return (
        landing_root
        / specification.source_id
        / f"run_id={run_id}"
        / expected_file_name
    )


def write_bytes_without_overwrite(payload: bytes, landing_path: Path) -> None:
    """Stage bytes locally, then copy them to a new immutable landing path."""

    if landing_path.exists():
        raise FileExistsError(f"Landing artifact already exists: {landing_path}")

    landing_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temporary_file:
            temporary_file.write(payload)
            temporary_path = Path(temporary_file.name)
        shutil.copyfile(temporary_path, landing_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def collect_remote_source(
    specification: SourceSpecification,
    run_id: str,
    landing_root: Path,
    request_parameter_overrides: dict[str, str] | None = None,
    timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
    fetcher: RemoteFetcher = fetch_http_response,
    collected_at: datetime | None = None,
) -> RemoteSourceCollectionResult:
    """Collect one bounded CKAN API response into an immutable landing path."""

    event_time = collected_at or datetime.now(timezone.utc)
    request_url: str | None = None

    try:
        if specification.acquisition_mode != "api":
            raise ValueError(
                "This slice supports acquisition_mode='api'; direct remote files "
                "will be added in the next adapter."
            )
        if specification.file_format != "json":
            raise ValueError("The bounded CKAN collector requires file_format='json'.")
        if not specification.url:
            raise ValueError("Remote collection requires a source URL.")

        request_parameters = dict(specification.params)
        request_parameters.update(request_parameter_overrides or {})
        request_url = build_request_url(specification.url, request_parameters)
        landing_path = build_landing_path(specification, landing_root, run_id)

        downloaded_response = fetcher(request_url, timeout_seconds)
        if not 200 <= downloaded_response.status_code < 300:
            raise ValueError(
                f"Remote request returned HTTP {downloaded_response.status_code}."
            )

        ckan_page = inspect_ckan_page(downloaded_response.payload)
        write_bytes_without_overwrite(downloaded_response.payload, landing_path)
        provenance = build_provenance(landing_path, landing_root)

        warnings: list[str] = []
        if ckan_page.page_record_count < ckan_page.total_record_count:
            warnings.append(
                "This is a bounded API page, not the complete source. Full CKAN "
                "pagination must complete before Bronze ingestion."
            )

        http_response = HTTPResponseInspection(
            request_url=request_url,
            status_code=downloaded_response.status_code,
            content_type=downloaded_response.content_type,
            response_size_bytes=len(downloaded_response.payload),
        )
        event = CollectionEvent(
            run_id=run_id,
            source_id=specification.source_id,
            acquisition_mode=specification.acquisition_mode,
            collection_method="direct_http",
            source_kind=specification.source_kind,
            origin_url=specification.url,
            request_url=request_url,
            landed_path=str(landing_path.resolve()),
            validated_at=event_time,
            retrieved_at=event_time,
            content_type=downloaded_response.content_type,
            file_size_bytes=provenance.file_size_bytes,
            sha256=provenance.sha256,
            record_count=ckan_page.page_record_count,
            status="succeeded",
            warnings=warnings,
        )
        return RemoteSourceCollectionResult(
            event=event,
            provenance=provenance,
            http_response=http_response,
            ckan_page=ckan_page,
        )
    except Exception as exc:
        event = CollectionEvent(
            run_id=run_id,
            source_id=specification.source_id,
            acquisition_mode=specification.acquisition_mode,
            collection_method="direct_http",
            source_kind=specification.source_kind,
            origin_url=specification.url,
            request_url=request_url,
            validated_at=event_time,
            status="failed",
            error_message=str(exc),
            warnings=[
                "The remote API source was not collected. Review error_message "
                "before continuing."
            ],
        )
        return RemoteSourceCollectionResult(event=event)
