from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


AcquisitionMode = Literal["local_file", "remote_file", "api"]
CollectionMethod = Literal["local_volume", "direct_http", "manual_upload"]
SourceKind = Literal["original_source", "derived_snapshot"]
SourceType = Literal["tabular", "geospatial"]
SourceFormat = Literal["csv", "geopackage", "json"]
CollectionStatus = Literal["succeeded", "failed", "reused_cached_copy"]


class SourceProvenance(BaseModel):
    filename: str
    relative_path: str
    absolute_path: str
    file_size_bytes: int
    modified_at: datetime
    sha256: str


class AccessMetadata(BaseModel):
    acquisition_mode: AcquisitionMode
    manifest_declared: bool = False
    origin_url: str | None = None
    request_url: str | None = None
    request_params: dict[str, str] = Field(default_factory=dict)
    content_type: str | None = None
    retrieved_at: datetime | None = None
    cache_path: str | None = None
    used_cached_copy: bool = False
    discovery_notes: list[str] = Field(default_factory=list)


class GeospatialMetadata(BaseModel):
    selected_layer: str | None = None
    available_layers: list[str] = Field(default_factory=list)
    geometry_column: str | None = None
    geometry_types: list[str] = Field(default_factory=list)


class CompatibilityHints(BaseModel):
    inferred_reference_year: int | None = None
    year_inference_basis: list[str] = Field(default_factory=list)
    candidate_join_columns: list[str] = Field(default_factory=list)
    candidate_spatial_unit_columns: list[str] = Field(default_factory=list)
    geometry_present: bool = False


class CollectionTransparency(BaseModel):
    loader_name: str
    decisions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SourceMetadata(BaseModel):
    source_name: str
    source_type: SourceType
    file_path: str
    file_format: SourceFormat
    row_count: int
    column_count: int
    columns: list[str]
    column_types: dict[str, str] = Field(default_factory=dict)
    crs: str | None = None
    collected_at: datetime
    access_metadata: AccessMetadata
    provenance: SourceProvenance
    geospatial_metadata: GeospatialMetadata | None = None
    compatibility_hints: CompatibilityHints = Field(default_factory=CompatibilityHints)
    transparency: CollectionTransparency


class CollectedSource(BaseModel):
    source_id: str
    source_metadata: SourceMetadata
    data: Any


class CollectedSourceSummary(BaseModel):
    source_id: str
    source_name: str
    source_type: SourceType
    file_format: SourceFormat
    acquisition_mode: AcquisitionMode
    row_count: int
    column_count: int
    inferred_reference_year: int | None = None
    selected_layer: str | None = None
    origin_url: str | None = None
    cache_path: str | None = None
    warning_count: int = 0


class SkippedCollectionItem(BaseModel):
    item_label: str
    item_origin: Literal["raw_scan", "manifest"]
    reason: str


class CollectionReport(BaseModel):
    status: Literal["completed", "warning"]
    collected_at: datetime
    raw_data_path: str
    manifest_path: str | None = None
    manifest_source_count: int = 0
    source_count: int
    sources: list[CollectedSourceSummary] = Field(default_factory=list)
    skipped_items: list[SkippedCollectionItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SourceSpecification(BaseModel):
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    acquisition_mode: AcquisitionMode
    source_kind: SourceKind = "original_source"
    source_type: SourceType
    file_format: SourceFormat
    source_name: str | None = None
    path: str | None = None
    manual_upload_path: str | None = None
    url: str | None = None
    params: dict[str, str] = Field(default_factory=dict)
    expected_file_name: str | None = None
    enabled: bool = True
    description: str | None = None
    discovery_notes: list[str] = Field(default_factory=list)
    upstream_sources: list[str] = Field(default_factory=list)
    transformation_reference: str | None = None
    csv_options: "CSVReadOptions | None" = None

    @model_validator(mode="after")
    def validate_source_location(self) -> "SourceSpecification":
        if self.acquisition_mode == "local_file" and not self.path:
            raise ValueError("Local file specifications require a 'path'.")

        if self.acquisition_mode in {"remote_file", "api"} and not self.url:
            raise ValueError("Remote file and API specifications require a 'url'.")

        if self.file_format == "geopackage" and self.source_type != "geospatial":
            raise ValueError("GeoPackage specifications must be marked as geospatial.")

        if self.acquisition_mode == "api" and self.source_type != "tabular":
            raise ValueError("API collection is currently bounded to tabular sources.")

        if (
            self.file_format == "json"
            and self.source_type == "geospatial"
            and self.acquisition_mode == "api"
        ):
            raise ValueError(
                "Geospatial JSON collection is currently bounded to local or remote-file sources, not API calls."
            )

        if self.source_kind == "derived_snapshot":
            if not self.upstream_sources:
                raise ValueError("Derived snapshots require at least one upstream source.")
            if not self.transformation_reference:
                raise ValueError("Derived snapshots require a transformation reference.")

        return self


class SourceManifest(BaseModel):
    sources: list[SourceSpecification] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_source_ids(self) -> "SourceManifest":
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Source manifest entries must have unique source IDs.")
        return self


def resolve_manifest_paths(
    manifest: SourceManifest,
    landing_root: str | Path,
) -> SourceManifest:
    """Resolve manifest-relative source paths inside the configured landing volume."""

    root = Path(landing_root)
    resolved_root = root.resolve()
    resolved_sources = []

    for source in manifest.sources:
        updates: dict[str, str] = {}
        for field_name in ("path", "manual_upload_path"):
            declared_path = getattr(source, field_name)
            if declared_path is None:
                continue

            path = Path(declared_path)
            if path.is_absolute():
                resolved_path = path.resolve()
            else:
                resolved_path = (root / path).resolve()
                try:
                    resolved_path.relative_to(resolved_root)
                except ValueError as exc:
                    raise ValueError(
                        f"Manifest path for {source.source_id} escapes the landing root: "
                        f"{declared_path}"
                    ) from exc
            updates[field_name] = str(resolved_path)

        resolved_sources.append(source.model_copy(update=updates))

    return manifest.model_copy(update={"sources": resolved_sources})


class CollectionEvent(BaseModel):
    """One auditable outcome for one source in one collection run."""

    run_id: str
    source_id: str
    acquisition_mode: AcquisitionMode
    collection_method: CollectionMethod
    source_kind: SourceKind
    origin_url: str | None = None
    request_url: str | None = None
    landed_path: str | None = None
    validated_at: datetime
    retrieved_at: datetime | None = None
    content_type: str | None = None
    file_size_bytes: int | None = None
    sha256: str | None = None
    record_count: int | None = None
    status: CollectionStatus
    used_cached_copy: bool = False
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_outcome(self) -> "CollectionEvent":
        if self.status == "succeeded" and (not self.landed_path or not self.sha256):
            raise ValueError("Successful collection events require a landed path and SHA-256.")

        if self.status == "reused_cached_copy" and not self.used_cached_copy:
            raise ValueError(
                "Cached-copy collection events must set used_cached_copy to true."
            )

        if self.status == "failed" and not self.error_message:
            raise ValueError("Failed collection events require an error message.")

        return self


class GeoPackageInspection(BaseModel):
    file_path: str
    selected_layer: str
    available_layers: list[str] = Field(default_factory=list)
    row_count: int
    column_count: int
    columns: list[str] = Field(default_factory=list)
    column_types: dict[str, str] = Field(default_factory=dict)
    crs: str | None = None
    srs_id: int | None = None
    geometry_column: str | None = None
    geometry_types: list[str] = Field(default_factory=list)


class CSVReadOptions(BaseModel):
    encoding: str = "utf-8"
    delimiter: str = ","
    header_row: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_delimiter(self) -> "CSVReadOptions":
        if len(self.delimiter) != 1:
            raise ValueError("CSV delimiter must contain exactly one character.")
        return self


class CSVInspection(BaseModel):
    file_path: str
    encoding: str
    delimiter: str
    header_row: int
    row_count: int
    column_count: int
    columns: list[str] = Field(default_factory=list)
    column_types: dict[str, str] = Field(default_factory=dict)
    empty_column_name_count: int = 0
    duplicate_column_name_count: int = 0


class LocalFileCollectionResult(BaseModel):
    event: CollectionEvent
    provenance: SourceProvenance | None = None
    geopackage: GeoPackageInspection | None = None
    csv: CSVInspection | None = None


class HTTPResponseInspection(BaseModel):
    request_url: str
    status_code: int
    content_type: str | None = None
    response_size_bytes: int


class JSONFileInspection(BaseModel):
    file_path: str
    top_level_keys: list[str] = Field(default_factory=list)
    root_type: str | None = None
    object_names: list[str] = Field(default_factory=list)
    record_count: int | None = None


class CKANPageInspection(BaseModel):
    success: bool
    total_record_count: int
    page_record_count: int
    limit: int
    offset: int


class RemoteSourceCollectionResult(BaseModel):
    event: CollectionEvent
    provenance: SourceProvenance | None = None
    http_response: HTTPResponseInspection | None = None
    ckan_page: CKANPageInspection | None = None
    json_file: JSONFileInspection | None = None
