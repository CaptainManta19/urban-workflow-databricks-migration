import re
from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


TECHNICAL_METADATA_COLUMNS = (
    "_source_id",
    "_source_kind",
    "_collection_method",
    "_source_path",
    "_source_sha256",
    "_bronze_run_id",
    "_ingested_at_utc",
)

INVALID_DELTA_COLUMN_NAME_PATTERN = re.compile(r"[ ,;{}()\n\t=]")


def find_invalid_delta_column_names(column_names: list[str]) -> list[str]:
    """Return source column names that need an explicit technical mapping."""

    return [
        column_name
        for column_name in column_names
        if not column_name
        or INVALID_DELTA_COLUMN_NAME_PATTERN.search(column_name) is not None
    ]


def apply_explicit_column_mapping(
    source_frame: DataFrame,
    column_mapping: dict[str, str],
) -> DataFrame:
    """Apply validated source-to-Bronze column names without changing values."""

    source_column_names = source_frame.columns
    unknown_source_columns = sorted(set(column_mapping) - set(source_column_names))
    if unknown_source_columns:
        raise ValueError(
            "Column mapping refers to fields that are absent from the source: "
            f"{unknown_source_columns}"
        )

    mapped_column_names = [
        column_mapping.get(column_name, column_name)
        for column_name in source_column_names
    ]
    invalid_target_names = find_invalid_delta_column_names(mapped_column_names)
    if invalid_target_names:
        raise ValueError(
            "Source fields still require explicit Delta-safe Bronze names: "
            f"{invalid_target_names}"
        )

    case_insensitive_name_groups: dict[str, list[str]] = {}
    for column_name in mapped_column_names:
        case_insensitive_name_groups.setdefault(column_name.casefold(), []).append(column_name)
    duplicate_target_names = sorted(
        {
            column_name
            for names in case_insensitive_name_groups.values()
            if len(names) > 1
            for column_name in names
        }
    )
    if duplicate_target_names:
        raise ValueError(
            "Column mapping produces duplicate Bronze names: "
            f"{duplicate_target_names}"
        )

    metadata_collisions = sorted(
        set(mapped_column_names) & set(TECHNICAL_METADATA_COLUMNS)
    )
    if metadata_collisions:
        raise ValueError(
            f"Mapped source fields collide with Bronze metadata: {metadata_collisions}"
        )

    return source_frame.select(
        *[
            F.col(f"`{source_name.replace('`', '``')}`").alias(
                column_mapping.get(source_name, source_name)
            )
            for source_name in source_column_names
        ]
    )


def add_bronze_metadata(
    source_frame: DataFrame,
    *,
    source_id: str,
    source_kind: str,
    collection_method: str,
    source_path: str,
    source_sha256: str,
    bronze_run_id: str,
    ingested_at_utc: datetime,
) -> DataFrame:
    """Add the shared row-level lineage contract to a source-shaped frame."""

    return (
        source_frame
        .withColumn("_source_id", F.lit(source_id))
        .withColumn("_source_kind", F.lit(source_kind))
        .withColumn("_collection_method", F.lit(collection_method))
        .withColumn("_source_path", F.lit(source_path))
        .withColumn("_source_sha256", F.lit(source_sha256))
        .withColumn("_bronze_run_id", F.lit(bronze_run_id))
        .withColumn("_ingested_at_utc", F.lit(ingested_at_utc))
    )


def write_snapshot_table(frame: DataFrame, table_name: str) -> None:
    """Replace one fixed-snapshot Delta table without appending duplicates."""

    (
        frame.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )


def validate_bronze_table(
    spark: SparkSession,
    *,
    table_name: str,
    expected_record_count: int,
    source_column_names: list[str],
) -> DataFrame:
    """Read a Bronze table back and enforce its shared persistence contract."""

    persisted_frame = spark.table(table_name)
    persisted_count = persisted_frame.count()
    if persisted_count != expected_record_count:
        raise ValueError(
            f"Persisted count mismatch for {table_name}: expected "
            f"{expected_record_count}, found {persisted_count}."
        )

    required_columns = [*source_column_names, *TECHNICAL_METADATA_COLUMNS]
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in persisted_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Persisted table {table_name} is missing required columns: "
            f"{missing_columns}"
        )

    metadata_null_counts = (
        persisted_frame
        .agg(
            *[
                F.sum(F.col(column_name).isNull().cast("int")).alias(column_name)
                for column_name in TECHNICAL_METADATA_COLUMNS
            ]
        )
        .first()
        .asDict()
    )
    metadata_with_nulls = {
        column_name: null_count
        for column_name, null_count in metadata_null_counts.items()
        if null_count
    }
    if metadata_with_nulls:
        raise ValueError(
            f"Bronze metadata contains nulls in {table_name}: {metadata_with_nulls}"
        )

    return persisted_frame
