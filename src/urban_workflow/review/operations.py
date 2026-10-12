"""Build and validate the final Databricks migration scorecard."""

from dataclasses import dataclass
from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType, TimestampType


@dataclass(frozen=True)
class ReviewCheck:
    group: str
    name: str
    passed: bool
    observed: str
    expected: str
    detail: str


def _exact_check(
    group: str,
    name: str,
    observed: int,
    expected: int,
    detail: str,
) -> ReviewCheck:
    return ReviewCheck(
        group=group,
        name=name,
        passed=observed == expected,
        observed=str(observed),
        expected=str(expected),
        detail=detail,
    )


def build_migration_review(
    spark: SparkSession,
    *,
    tables: dict[str, str],
    silhouette_score: float,
    review_run_id: str,
    reviewed_at_utc: datetime,
) -> DataFrame:
    """Collect the accepted layer and model contracts into one scorecard."""

    expected_counts = {
        "bronze_emvs": 22,
        "bronze_population": 1_071,
        "bronze_indicators": 116_930,
        "bronze_boundaries": 21,
        "bronze_grid": 10_100,
        "silver_districts": 21,
        "silver_emvs": 21,
        "silver_population": 1_071,
        "silver_indicators": 116_930,
        "silver_boundaries": 21,
        "gold_grid": 10_100,
        "gold_districts": 21,
        "grid_typology": 10_100,
        "cluster_profiles": 3,
        "district_cluster_mix": 21,
        "district_anomalies": 21,
    }
    checks = []
    for alias, expected_count in expected_counts.items():
        observed_count = spark.table(tables[alias]).count()
        checks.append(
            _exact_check(
                "table_counts",
                f"{alias}_row_count",
                observed_count,
                expected_count,
                f"Persisted row count for {tables[alias]}.",
            )
        )

    gold_grid = spark.table(tables["gold_grid"])
    grid_quality = gold_grid.agg(
        F.sum(
            (F.col("district_mapping_status") == "canonical_match").cast("int")
        ).alias("mapped_rows"),
        F.sum(
            (
                F.col("district_mapping_status") == "source_district_missing"
            ).cast("int")
        ).alias("source_missing_rows"),
        F.sum(F.col("building_height_mean_m").isNull().cast("int")).alias(
            "height_null_rows"
        ),
        F.sum(F.col("rent_median_eur_m2_2023").isNull().cast("int")).alias(
            "rent_null_rows"
        ),
        F.sum(F.col("emvs_housing_units_total").isNull().cast("int")).alias(
            "emvs_null_rows"
        ),
    ).first().asDict()
    grid_expectations = {
        "mapped_rows": 9_667,
        "source_missing_rows": 433,
        "height_null_rows": 5_589,
        "rent_null_rows": 787,
        "emvs_null_rows": 433,
    }
    for name, expected in grid_expectations.items():
        checks.append(
            _exact_check(
                "gold_grid_quality",
                name,
                int(grid_quality[name]),
                expected,
                "Accepted source coverage remains explicit in Gold.",
            )
        )

    grid_typology = spark.table(tables["grid_typology"])
    cluster_profiles = spark.table(tables["cluster_profiles"])
    eligible_rows = grid_typology.filter("cluster_eligible").count()
    profile_cells = int(
        cluster_profiles.agg(F.sum("cell_count").alias("value")).first()["value"]
    )
    checks.append(
        _exact_check(
            "kmeans",
            "eligible_profile_cell_reconciliation",
            profile_cells,
            eligible_rows,
            "Cluster-profile cells must equal eligible grid predictions.",
        )
    )
    profile_share_sum = float(
        cluster_profiles.agg(
            F.sum("share_of_eligible_cells").alias("value")
        ).first()["value"]
    )
    checks.append(
        ReviewCheck(
            group="kmeans",
            name="cluster_profile_share_sum",
            passed=abs(profile_share_sum - 1.0) <= 1e-9,
            observed=f"{profile_share_sum:.12f}",
            expected="1.0 ± 1e-9",
            detail="The three cluster profile shares must exhaust eligible cells.",
        )
    )
    district_mix = spark.table(tables["district_cluster_mix"])
    share_sum = sum(
        F.col(f"cluster_share_cluster_{index}") for index in range(3)
    )
    district_share_failures = district_mix.filter(
        F.abs(share_sum - F.lit(1.0)) > F.lit(1e-9)
    ).count()
    checks.append(
        _exact_check(
            "kmeans",
            "district_cluster_share_failures",
            district_share_failures,
            0,
            "Every district cluster-share vector must sum to one.",
        )
    )

    anomalies = spark.table(tables["district_anomalies"])
    anomaly_quality = anomalies.agg(
        F.sum(F.col("anomaly_flag").cast("int")).alias("flagged_rows"),
        F.countDistinct("anomaly_rank").alias("distinct_ranks"),
        F.sum(
            (
                (F.size("anomaly_top_features") != 3)
                | (F.size("anomaly_interpretation_notes") != 3)
            ).cast("int")
        ).alias("explanation_failures"),
    ).first().asDict()
    checks.extend(
        [
            _exact_check(
                "isolation_forest",
                "exploratory_flag_count",
                int(anomaly_quality["flagged_rows"]),
                2,
                "Accepted fixed-snapshot IsolationForest flag count.",
            ),
            _exact_check(
                "isolation_forest",
                "distinct_anomaly_ranks",
                int(anomaly_quality["distinct_ranks"]),
                21,
                "Every district must have one deterministic review rank.",
            ),
            _exact_check(
                "isolation_forest",
                "anomaly_explanation_failures",
                int(anomaly_quality["explanation_failures"]),
                0,
                "Every district must retain three descriptive explanation features.",
            ),
        ]
    )
    checks.append(
        ReviewCheck(
            group="mlflow",
            name="kmeans_silhouette_range",
            passed=-1.0 <= silhouette_score <= 1.0,
            observed=f"{silhouette_score:.12f}",
            expected="between -1.0 and 1.0",
            detail="Silhouette is retrieved from the accepted KMeans MLflow run.",
        )
    )

    schema = StructType(
        [
            StructField("check_group", StringType(), nullable=False),
            StructField("check_name", StringType(), nullable=False),
            StructField("status", StringType(), nullable=False),
            StructField("observed_value", StringType(), nullable=False),
            StructField("expected_value", StringType(), nullable=False),
            StructField("detail", StringType(), nullable=False),
            StructField("review_run_id", StringType(), nullable=False),
            StructField("reviewed_at_utc", TimestampType(), nullable=False),
        ]
    )
    return spark.createDataFrame(
        [
            {
                "check_group": check.group,
                "check_name": check.name,
                "status": "passed" if check.passed else "failed",
                "observed_value": check.observed,
                "expected_value": check.expected,
                "detail": check.detail,
                "review_run_id": review_run_id,
                "reviewed_at_utc": reviewed_at_utc,
            }
            for check in checks
        ],
        schema=schema,
    )


def validate_migration_review(review: DataFrame) -> dict[str, int]:
    """Require the complete 28-check scorecard to pass."""

    summary = review.agg(
        F.count("*").alias("check_count"),
        F.countDistinct("check_name").alias("distinct_check_names"),
        F.sum((F.col("status") == "passed").cast("int")).alias("passed_checks"),
        F.sum((F.col("status") == "failed").cast("int")).alias("failed_checks"),
    ).first().asDict()
    expected = {
        "check_count": 28,
        "distinct_check_names": 28,
        "passed_checks": 28,
        "failed_checks": 0,
    }
    failures = {
        key: (value, summary.get(key))
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if failures:
        failed_rows = [row.asDict() for row in review.filter("status = 'failed'").collect()]
        raise ValueError(
            f"Migration review failed: summary={failures}, checks={failed_rows}"
        )
    return {key: int(value) for key, value in summary.items()}


def write_snapshot_table(frame: DataFrame, table_name: str) -> None:
    """Replace the deterministic review scorecard."""

    (
        frame.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
