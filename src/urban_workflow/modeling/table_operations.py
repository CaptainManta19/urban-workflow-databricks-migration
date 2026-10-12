"""Spark persistence contracts for KMeans typology outputs."""

from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from urban_workflow.modeling.anomaly_detection import (
    ANOMALY_FEATURE_COLUMNS,
    IsolationForestArtifacts,
)
from urban_workflow.modeling.clustering import CLUSTER_COUNT, KMeansArtifacts


def build_typology_output_frames(
    spark: SparkSession,
    source_grid: DataFrame,
    artifacts: KMeansArtifacts,
    *,
    model_run_id: str,
    mlflow_run_id: str,
    trained_at_utc: datetime,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Convert model artifacts to lineage-rich Spark frames."""

    source_gold_run_ids = [
        row["_gold_run_id"]
        for row in source_grid.select("_gold_run_id").distinct().collect()
    ]
    if len(source_gold_run_ids) != 1 or source_gold_run_ids[0] is None:
        raise ValueError(
            "KMeans requires exactly one non-null source Gold run ID; found "
            f"{source_gold_run_ids}."
        )
    source_gold_run_id = source_gold_run_ids[0]

    prediction_schema = StructType(
        [
            StructField("cell_id", LongType(), nullable=False),
            StructField("cluster_eligible", BooleanType(), nullable=False),
            StructField("cluster_ineligibility_reason", StringType(), nullable=True),
            StructField("cluster_label", StringType(), nullable=True),
            StructField("cluster_distance_to_centroid", DoubleType(), nullable=True),
        ]
    )
    prediction_records = []
    for record in artifacts.grid_predictions.to_dict("records"):
        reason = record["cluster_ineligibility_reason"]
        label = record["cluster_label"]
        distance = record["cluster_distance_to_centroid"]
        prediction_records.append(
            {
                "cell_id": int(record["cell_id"]),
                "cluster_eligible": bool(record["cluster_eligible"]),
                "cluster_ineligibility_reason": (
                    None if reason is None or reason != reason else str(reason)
                ),
                "cluster_label": (
                    None if label is None or label != label else str(label)
                ),
                "cluster_distance_to_centroid": (
                    None if distance is None or distance != distance else float(distance)
                ),
            }
        )
    predictions = spark.createDataFrame(prediction_records, schema=prediction_schema)
    grid_typology = source_grid.join(predictions, on="cell_id", how="inner")

    profile_schema = StructType(
        [
            StructField("cluster_label", StringType(), nullable=False),
            StructField("cell_count", LongType(), nullable=False),
            StructField("share_of_eligible_cells", DoubleType(), nullable=False),
            StructField("dominant_land_use_class", StringType(), nullable=False),
            StructField("mean_building_height_mean_m", DoubleType(), nullable=False),
            StructField("mean_building_height_max_m", DoubleType(), nullable=False),
            StructField(
                "mean_public_transport_stop_count", DoubleType(), nullable=False
            ),
            StructField("narrative_label", StringType(), nullable=False),
        ]
    )
    profiles = spark.createDataFrame(
        artifacts.cluster_profiles.to_dict("records"),
        schema=profile_schema,
    )

    mix_schema = StructType(
        [
            StructField("district_code", StringType(), nullable=False),
            StructField("district_name", StringType(), nullable=False),
            *[
                StructField(
                    f"cluster_share_cluster_{index}",
                    DoubleType(),
                    nullable=False,
                )
                for index in range(CLUSTER_COUNT)
            ],
            StructField("dominant_cluster_label", StringType(), nullable=False),
        ]
    )
    district_mix = spark.createDataFrame(
        artifacts.district_cluster_mix.to_dict("records"),
        schema=mix_schema,
    )

    def add_model_metadata(frame: DataFrame) -> DataFrame:
        return (
            frame
            .withColumn("model_name", F.lit("kmeans"))
            .withColumn("model_run_id", F.lit(model_run_id))
            .withColumn("mlflow_run_id", F.lit(mlflow_run_id))
            .withColumn("model_trained_at_utc", F.lit(trained_at_utc))
            .withColumn("cluster_count", F.lit(CLUSTER_COUNT))
            .withColumn("source_gold_run_id", F.lit(source_gold_run_id))
        )

    return tuple(
        add_model_metadata(frame)
        for frame in (grid_typology, profiles, district_mix)
    )


def validate_typology_outputs(
    source_grid: DataFrame,
    grid_typology: DataFrame,
    profiles: DataFrame,
    district_mix: DataFrame,
) -> dict[str, int | float]:
    """Validate model eligibility, predictions, profiles, and district shares."""

    expected_eligible = source_grid.filter(
        (F.col("district_mapping_status") == "canonical_match")
        & F.col("land_use_class_simplified").isNotNull()
        & F.col("building_height_mean_m").isNotNull()
        & F.col("building_height_max_m").isNotNull()
        & F.col("public_transport_stop_count").isNotNull()
    ).count()
    grid_summary = grid_typology.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("cell_id").alias("distinct_cell_ids"),
        F.sum(F.col("cluster_eligible").cast("int")).alias("eligible_rows"),
        F.sum(
            (
                F.col("cluster_eligible")
                & (
                    F.col("cluster_label").isNull()
                    | F.col("cluster_distance_to_centroid").isNull()
                    | (F.col("cluster_distance_to_centroid") < 0)
                )
            ).cast("int")
        ).alias("invalid_eligible_predictions"),
        F.sum(
            (
                (~F.col("cluster_eligible"))
                & (
                    F.col("cluster_label").isNotNull()
                    | F.col("cluster_distance_to_centroid").isNotNull()
                    | F.col("cluster_ineligibility_reason").isNull()
                )
            ).cast("int")
        ).alias("invalid_ineligible_predictions"),
        F.countDistinct("cluster_label").alias("predicted_cluster_count"),
        F.sum(
            (
                F.col("model_run_id").isNull()
                | F.col("mlflow_run_id").isNull()
                | F.col("model_trained_at_utc").isNull()
                | F.col("source_gold_run_id").isNull()
                | (F.col("model_name") != "kmeans")
                | (F.col("cluster_count") != CLUSTER_COUNT)
            ).cast("int")
        ).alias("grid_metadata_failures"),
    ).first().asDict()
    profile_summary = profiles.agg(
        F.count("*").alias("profile_count"),
        F.countDistinct("cluster_label").alias("profile_cluster_count"),
        F.sum("cell_count").alias("profile_cell_count"),
        F.sum("share_of_eligible_cells").alias("profile_share_sum"),
        F.sum(
            (
                F.col("model_run_id").isNull()
                | F.col("mlflow_run_id").isNull()
                | F.col("source_gold_run_id").isNull()
            ).cast("int")
        ).alias("profile_metadata_failures"),
    ).first().asDict()
    share_columns = [
        F.col(f"cluster_share_cluster_{index}")
        for index in range(CLUSTER_COUNT)
    ]
    mix_summary = district_mix.agg(
        F.count("*").alias("district_mix_rows"),
        F.countDistinct("district_code").alias("district_mix_codes"),
        F.sum(
            (
                F.abs(sum(share_columns) - F.lit(1.0)) > F.lit(1e-9)
            ).cast("int")
        ).alias("district_share_failures"),
        F.sum(F.col("dominant_cluster_label").isNull().cast("int")).alias(
            "missing_dominant_clusters"
        ),
        F.sum(
            (
                F.col("model_run_id").isNull()
                | F.col("mlflow_run_id").isNull()
                | F.col("source_gold_run_id").isNull()
            ).cast("int")
        ).alias("mix_metadata_failures"),
    ).first().asDict()
    failures: dict[str, tuple[object, object]] = {}
    expected_values = {
        "row_count": 10_100,
        "distinct_cell_ids": 10_100,
        "eligible_rows": expected_eligible,
        "invalid_eligible_predictions": 0,
        "invalid_ineligible_predictions": 0,
        "predicted_cluster_count": CLUSTER_COUNT,
        "grid_metadata_failures": 0,
        "profile_count": CLUSTER_COUNT,
        "profile_cluster_count": CLUSTER_COUNT,
        "profile_cell_count": expected_eligible,
        "profile_metadata_failures": 0,
        "district_mix_rows": 21,
        "district_mix_codes": 21,
        "district_share_failures": 0,
        "missing_dominant_clusters": 0,
        "mix_metadata_failures": 0,
    }
    combined = {**grid_summary, **profile_summary, **mix_summary}
    for key, expected in expected_values.items():
        if combined.get(key) != expected:
            failures[key] = (expected, combined.get(key))
    if abs(float(profile_summary["profile_share_sum"]) - 1.0) > 1e-9:
        failures["profile_share_sum"] = (1.0, profile_summary["profile_share_sum"])
    if failures:
        raise ValueError(f"KMeans typology validation failed: {failures}")
    return {
        **{
            key: int(value)
            for key, value in combined.items()
            if key != "profile_share_sum"
        },
        "profile_share_sum": float(profile_summary["profile_share_sum"]),
    }


def build_anomaly_output_frame(
    spark: SparkSession,
    district_features: DataFrame,
    district_cluster_mix: DataFrame,
    artifacts: IsolationForestArtifacts,
    *,
    model_run_id: str,
    mlflow_run_id: str,
    trained_at_utc: datetime,
) -> DataFrame:
    """Attach IsolationForest predictions and full source/model lineage."""

    district_gold_run_ids = [
        row["_gold_run_id"]
        for row in district_features.select("_gold_run_id").distinct().collect()
    ]
    kmeans_run_ids = [
        row["model_run_id"]
        for row in district_cluster_mix.select("model_run_id").distinct().collect()
    ]
    if len(district_gold_run_ids) != 1 or district_gold_run_ids[0] is None:
        raise ValueError(
            "IsolationForest requires one district Gold run ID; found "
            f"{district_gold_run_ids}."
        )
    if len(kmeans_run_ids) != 1 or kmeans_run_ids[0] is None:
        raise ValueError(
            "IsolationForest requires one KMeans run ID; found "
            f"{kmeans_run_ids}."
        )

    schema = StructType(
        [
            StructField("district_code", StringType(), nullable=False),
            StructField("anomaly_eligible", BooleanType(), nullable=False),
            StructField("anomaly_score", DoubleType(), nullable=True),
            StructField("anomaly_flag", BooleanType(), nullable=True),
            StructField("anomaly_rank", IntegerType(), nullable=True),
            StructField(
                "anomaly_top_features",
                ArrayType(StringType(), containsNull=False),
                nullable=False,
            ),
            StructField(
                "anomaly_interpretation_notes",
                ArrayType(StringType(), containsNull=False),
                nullable=False,
            ),
        ]
    )
    records = []
    for record in artifacts.district_predictions.to_dict("records"):
        score = record["anomaly_score"]
        flag = record["anomaly_flag"]
        rank = record["anomaly_rank"]
        records.append(
            {
                "district_code": str(record["district_code"]),
                "anomaly_eligible": bool(record["anomaly_eligible"]),
                "anomaly_score": (
                    None if score is None or score != score else float(score)
                ),
                "anomaly_flag": (
                    None if flag is None or flag != flag else bool(flag)
                ),
                "anomaly_rank": (
                    None if rank is None or rank != rank else int(rank)
                ),
                "anomaly_top_features": list(record["anomaly_top_features"]),
                "anomaly_interpretation_notes": list(
                    record["anomaly_interpretation_notes"]
                ),
            }
        )
    predictions = spark.createDataFrame(records, schema=schema)
    cluster_shares = district_cluster_mix.select(
        "district_code",
        *[
            f"cluster_share_cluster_{index}"
            for index in range(CLUSTER_COUNT)
        ],
        "dominant_cluster_label",
    )
    return (
        district_features
        .join(cluster_shares, on="district_code", how="left")
        .join(predictions, on="district_code", how="inner")
        .withColumn("model_name", F.lit("isolation_forest"))
        .withColumn("model_run_id", F.lit(model_run_id))
        .withColumn("mlflow_run_id", F.lit(mlflow_run_id))
        .withColumn("model_trained_at_utc", F.lit(trained_at_utc))
        .withColumn(
            "source_district_gold_run_id",
            F.lit(district_gold_run_ids[0]),
        )
        .withColumn("source_kmeans_model_run_id", F.lit(kmeans_run_ids[0]))
    )


def validate_anomaly_output(
    district_features: DataFrame,
    district_anomalies: DataFrame,
) -> dict[str, int]:
    """Validate district scores, ranks, explanations, and lineage."""

    feature_present = None
    for feature_name in ANOMALY_FEATURE_COLUMNS:
        current = F.col(feature_name).isNotNull()
        feature_present = current if feature_present is None else feature_present | current
    expected_eligible = district_anomalies.filter(feature_present).count()
    summary = district_anomalies.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("district_code").alias("distinct_district_codes"),
        F.sum(F.col("anomaly_eligible").cast("int")).alias("eligible_rows"),
        F.sum(F.coalesce(F.col("anomaly_flag"), F.lit(False)).cast("int")).alias(
            "flagged_rows"
        ),
        F.countDistinct("anomaly_rank").alias("distinct_ranks"),
        F.min("anomaly_rank").alias("minimum_rank"),
        F.max("anomaly_rank").alias("maximum_rank"),
        F.sum(
            (
                F.col("anomaly_eligible")
                & (
                    F.col("anomaly_score").isNull()
                    | F.isnan("anomaly_score")
                    | F.col("anomaly_rank").isNull()
                    | (F.size("anomaly_top_features") != 3)
                    | (F.size("anomaly_interpretation_notes") != 3)
                )
            ).cast("int")
        ).alias("invalid_eligible_rows"),
        F.sum(
            (
                (~F.col("anomaly_eligible"))
                & (
                    F.col("anomaly_score").isNotNull()
                    | F.col("anomaly_flag").isNotNull()
                    | F.col("anomaly_rank").isNotNull()
                )
            ).cast("int")
        ).alias("invalid_ineligible_rows"),
        F.sum(
            (
                F.col("model_run_id").isNull()
                | F.col("mlflow_run_id").isNull()
                | F.col("model_trained_at_utc").isNull()
                | F.col("source_district_gold_run_id").isNull()
                | F.col("source_kmeans_model_run_id").isNull()
                | (F.col("model_name") != "isolation_forest")
            ).cast("int")
        ).alias("metadata_failures"),
    ).first().asDict()
    expected = {
        "row_count": 21,
        "distinct_district_codes": 21,
        "eligible_rows": expected_eligible,
        "distinct_ranks": expected_eligible,
        "minimum_rank": 1,
        "maximum_rank": expected_eligible,
        "invalid_eligible_rows": 0,
        "invalid_ineligible_rows": 0,
        "metadata_failures": 0,
    }
    failures = {
        key: (value, summary.get(key))
        for key, value in expected.items()
        if summary.get(key) != value
    }
    flagged_rows = int(summary["flagged_rows"])
    if flagged_rows < 1 or flagged_rows > 4:
        failures["flagged_rows"] = ("between 1 and 4", flagged_rows)
    if district_features.count() != 21:
        failures["source_district_rows"] = (21, district_features.count())
    if failures:
        raise ValueError(f"IsolationForest validation failed: {failures}")
    return {key: int(value) for key, value in summary.items()}


def write_snapshot_table(frame: DataFrame, table_name: str) -> None:
    """Replace one deterministic modeling output table."""

    (
        frame.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
