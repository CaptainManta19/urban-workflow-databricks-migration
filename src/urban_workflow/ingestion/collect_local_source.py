import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from urban_workflow.ingestion.provenance import (
    build_provenance,
    ensure_file_is_materialised,
)
from urban_workflow.ingestion.schemas import (
    CSVInspection,
    CSVReadOptions,
    CollectionEvent,
    GeoPackageInspection,
    LocalFileCollectionResult,
    SourceSpecification,
)


GEOPACKAGE_CONTENT_TYPE = "application/geopackage+sqlite3"
CSV_CONTENT_TYPE = "text/csv"


def _quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier after it came from GeoPackage metadata."""

    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _open_sqlite_read_only(file_path: Path) -> sqlite3.Connection:
    encoded_path = quote(str(file_path.resolve()), safe="/")
    return sqlite3.connect(f"file:{encoded_path}?mode=ro", uri=True)


def inspect_geopackage(
    file_path: Path,
    selected_layer: str | None = None,
) -> GeoPackageInspection:
    """Inspect GeoPackage catalog metadata without loading its geometry rows."""

    if not file_path.is_file():
        raise FileNotFoundError(f"GeoPackage does not exist: {file_path}")

    ensure_file_is_materialised(file_path)

    try:
        with _open_sqlite_read_only(file_path) as connection:
            layer_rows = connection.execute(
                """
                SELECT
                    contents.table_name,
                    geometry.column_name,
                    geometry.geometry_type_name,
                    geometry.srs_id
                FROM gpkg_contents AS contents
                INNER JOIN gpkg_geometry_columns AS geometry
                    ON contents.table_name = geometry.table_name
                WHERE contents.data_type = 'features'
                ORDER BY contents.table_name
                """
            ).fetchall()

            if not layer_rows:
                raise ValueError("GeoPackage does not contain a feature layer.")

            available_layers = [str(row[0]) for row in layer_rows]
            layer_name = selected_layer or available_layers[0]
            if layer_name not in available_layers:
                raise ValueError(
                    f"Requested layer '{layer_name}' was not found. "
                    f"Available layers: {available_layers}"
                )

            layer_metadata = next(row for row in layer_rows if row[0] == layer_name)
            geometry_column = str(layer_metadata[1])
            geometry_type = str(layer_metadata[2])
            srs_id = int(layer_metadata[3])

            quoted_layer = _quote_identifier(layer_name)
            row_count = int(
                connection.execute(f"SELECT COUNT(*) FROM {quoted_layer}").fetchone()[0]
            )
            table_info = connection.execute(
                f"PRAGMA table_info({quoted_layer})"
            ).fetchall()

            columns = [str(row[1]) for row in table_info]
            column_types = {str(row[1]): str(row[2]) for row in table_info}

            srs_row = connection.execute(
                """
                SELECT organization, organization_coordsys_id
                FROM gpkg_spatial_ref_sys
                WHERE srs_id = ?
                """,
                (srs_id,),
            ).fetchone()

            crs = None
            if srs_row and srs_row[0] and srs_row[1] is not None:
                crs = f"{srs_row[0]}:{srs_row[1]}"
    except sqlite3.DatabaseError as exc:
        raise ValueError(
            "File could not be read as a valid GeoPackage SQLite database."
        ) from exc

    return GeoPackageInspection(
        file_path=str(file_path.resolve()),
        selected_layer=layer_name,
        available_layers=available_layers,
        row_count=row_count,
        column_count=len(columns),
        columns=columns,
        column_types=column_types,
        crs=crs,
        srs_id=srs_id,
        geometry_column=geometry_column,
        geometry_types=[geometry_type],
    )


def inspect_csv(
    file_path: Path,
    options: CSVReadOptions,
) -> CSVInspection:
    """Inspect CSV structure without applying business cleaning or type inference."""

    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {file_path}")

    ensure_file_is_materialised(file_path)

    try:
        with file_path.open("r", encoding=options.encoding, newline="") as file_handle:
            reader = csv.reader(file_handle, delimiter=options.delimiter)

            for skipped_row_number in range(options.header_row):
                try:
                    next(reader)
                except StopIteration as exc:
                    raise ValueError(
                        f"CSV ended before configured header row {options.header_row}."
                    ) from exc

            try:
                columns = [column.strip() for column in next(reader)]
            except StopIteration as exc:
                raise ValueError("CSV does not contain the configured header row.") from exc

            if not columns or not any(columns):
                raise ValueError("CSV header does not contain a non-empty column name.")

            row_count = sum(1 for row in reader if any(value.strip() for value in row))
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"CSV could not be decoded using encoding '{options.encoding}'."
        ) from exc

    return CSVInspection(
        file_path=str(file_path.resolve()),
        encoding=options.encoding,
        delimiter=options.delimiter,
        header_row=options.header_row,
        row_count=row_count,
        column_count=len(columns),
        columns=columns,
        column_types={column: "string" for column in columns if column},
        empty_column_name_count=sum(not column for column in columns),
        duplicate_column_name_count=len(columns) - len(set(columns)),
    )


def collect_local_source(
    specification: SourceSpecification,
    run_id: str,
    landing_root: Path,
    collected_at: datetime | None = None,
) -> LocalFileCollectionResult:
    """Dispatch one manifest-declared local file to its format adapter."""

    event_time = collected_at or datetime.now(timezone.utc)
    file_path = Path(specification.path) if specification.path else None

    try:
        if specification.acquisition_mode != "local_file":
            raise ValueError(
                "Local collection requires acquisition_mode='local_file'."
            )
        if file_path is None:
            raise ValueError("Local collection requires a file path.")

        warnings: list[str] = []
        geopackage_inspection = None
        csv_inspection = None

        if specification.file_format == "geopackage":
            geopackage_inspection = inspect_geopackage(file_path)
            record_count = geopackage_inspection.row_count
            content_type = GEOPACKAGE_CONTENT_TYPE
            if len(geopackage_inspection.available_layers) > 1:
                warnings.append(
                    "Multiple feature layers were found. The first layer was selected "
                    "for this prototype run."
                )
        elif specification.file_format == "csv":
            csv_options = specification.csv_options or CSVReadOptions()
            csv_inspection = inspect_csv(file_path, csv_options)
            record_count = csv_inspection.row_count
            content_type = CSV_CONTENT_TYPE
            if csv_inspection.empty_column_name_count:
                warnings.append(
                    f"CSV header contains {csv_inspection.empty_column_name_count} "
                    "empty column name(s). Preserve them in landing and resolve them "
                    "during Bronze or Silver processing."
                )
            if csv_inspection.duplicate_column_name_count:
                warnings.append(
                    f"CSV header contains {csv_inspection.duplicate_column_name_count} "
                    "duplicate column name occurrence(s)."
                )
        else:
            raise ValueError(
                f"Local collection does not yet support file format "
                f"'{specification.file_format}'."
            )

        provenance = build_provenance(file_path, landing_root)
        event = CollectionEvent(
            run_id=run_id,
            source_id=specification.source_id,
            acquisition_mode=specification.acquisition_mode,
            collection_method="local_volume",
            source_kind=specification.source_kind,
            landed_path=str(file_path.resolve()),
            validated_at=event_time,
            content_type=content_type,
            file_size_bytes=provenance.file_size_bytes,
            sha256=provenance.sha256,
            record_count=record_count,
            status="succeeded",
            warnings=warnings,
        )
        return LocalFileCollectionResult(
            event=event,
            provenance=provenance,
            geopackage=geopackage_inspection,
            csv=csv_inspection,
        )
    except Exception as exc:
        event = CollectionEvent(
            run_id=run_id,
            source_id=specification.source_id,
            acquisition_mode=specification.acquisition_mode,
            collection_method="local_volume",
            source_kind=specification.source_kind,
            landed_path=str(file_path) if file_path else None,
            validated_at=event_time,
            status="failed",
            error_message=str(exc),
            warnings=[
                "The local source was not collected. Review error_message before continuing."
            ],
        )
        return LocalFileCollectionResult(event=event)
