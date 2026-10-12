"""Spark transformations and acceptance checks for Gold feature tables."""

from datetime import datetime

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from urban_workflow.bronze.table_operations import TECHNICAL_METADATA_COLUMNS


GOLD_METADATA_COLUMNS = (
    "_gold_run_id",
    "_gold_processed_at_utc",
)

GRID_FEATURE_COLUMNS = (
    "land_use_code",
    "land_use_class",
    "land_use_class_simplified",
    "building_height_mean_m",
    "building_height_max_m",
    "public_transport_stop_count",
    "has_good_public_transport_access",
    "rent_median_eur_m2_2023",
    "emvs_housing_units_total",
)

GREEN_LIKE_CLASSES = (
    "Green urban areas",
    "Herbaceous vegetation associations (natural grassland, moors...)",
    "Pastures",
    "Arable land (annual crops)",
)
RESIDENTIAL_CLASSES = (
    "Continuous urban fabric (S.L. : > 80%)",
    "Discontinuous dense urban fabric (S.L. : 50% -  80%)",
)
INDUSTRIAL_CLASSES = (
    "Industrial, commercial, public, military and private units",
)
DENSE_URBAN_CLASSES = RESIDENTIAL_CLASSES

DISTRICT_INDICATOR_FEATURES = {
    "green_area_ha": "Superficie de zonas verdes y parques de distrito (ha.)",
    "green_area_per_10000": (
        "Superficie de zonas verdes y parques de distrito (ha.) entre número "
        "de habitantes *10.000"
    ),
    "income_per_person": "Renta media disponible por persona",
    "household_income": "Renta neta media anual de los hogares (Urban Audit)",
    "unemployment_total": (
        "Paro registrado (número de personas registradas en SEPE en febrero)"
    ),
    "unemployment_rate": "Tasa absoluta de paro registrado (febrero)",
    "vulnerability_index": "Índice de vulnerabilidad territorial agregado",
    "vulnerability_employment": "Índice de vulnerabilidad economía y empleo",
}

DISTRICT_FEATURE_COLUMNS = (
    "area_m2",
    "area_km2",
    "population_reference_date",
    "population_reference_year",
    "population_total",
    "population_male",
    "population_female",
    "population_density_per_km2",
    "housing_reporting_start_date",
    "housing_reporting_end_date",
    "housing_units_total",
    "housing_units_regulation",
    "housing_units_other_programs",
    "housing_units_per_1000_residents",
    "grid_cell_count",
    "grid_height_observed_cell_count",
    "grid_height_coverage_share",
    "grid_building_height_mean_m_avg",
    "grid_building_height_max_m_avg",
    "grid_public_transport_stop_count_avg",
    "grid_good_public_transport_access_share",
    "grid_green_like_share",
    "grid_residential_share",
    "grid_industrial_share",
    "grid_dense_urban_share",
    "grid_rent_observed_cell_count",
    "grid_rent_coverage_share",
    "grid_rent_median_eur_m2_2023",
    *DISTRICT_INDICATOR_FEATURES.keys(),
    *[f"{name}_year" for name in DISTRICT_INDICATOR_FEATURES],
    "has_green_data",
    "has_economy_data",
    "has_employment_data",
    "has_vulnerability_data",
    "district_indicator_coverage_share",
    "district_feature_coverage_score",
)


def normalized_district_key(column: Column) -> Column:
    """Apply the established Silver district-key rule to the grid labels."""

    return F.regexp_replace(
        F.translate(F.upper(F.trim(column)), "ÁÉÍÓÚÜÑ", "AEIOUUN"),
        "[^A-Z0-9]",
        "",
    )


def add_gold_metadata(
    frame: DataFrame,
    *,
    gold_run_id: str,
    processed_at_utc: datetime,
) -> DataFrame:
    """Attach the processing identity used by deterministic Gold snapshots."""

    return (
        frame
        .withColumn("_gold_run_id", F.lit(gold_run_id))
        .withColumn("_gold_processed_at_utc", F.lit(processed_at_utc))
    )


def build_grid_features(
    bronze_grid: DataFrame,
    district_reference: DataFrame,
) -> DataFrame:
    """Create one modeling-ready row per preserved legacy grid cell."""

    staged = (
        bronze_grid
        .withColumn("source_feature_id", F.col("fid").cast("long"))
        .withColumn("cell_id", F.col("cell_id").cast("long"))
        .withColumn("district_name_source", F.col("district_name"))
        .withColumn(
            "district_name_key_source",
            normalized_district_key(F.col("district_name")),
        )
        .drop("district_name")
        .withColumnRenamed("lu_2018_code", "land_use_code")
        .withColumnRenamed("lu_2018_class", "land_use_class")
        .withColumnRenamed(
            "lu_2018_class_simplified",
            "land_use_class_simplified",
        )
        .withColumnRenamed("height_mean", "building_height_mean_m")
        .withColumnRenamed("height_max", "building_height_max_m")
        .withColumnRenamed("pt_stop_count", "public_transport_stop_count")
        .withColumnRenamed(
            "pt_access_good",
            "has_good_public_transport_access",
        )
        .withColumnRenamed(
            "rent_median_m2_2023",
            "rent_median_eur_m2_2023",
        )
        .withColumnRenamed("emvs_units_total", "emvs_housing_units_total")
    )
    canonical_districts = district_reference.select(
        "district_code",
        "district_number",
        "district_name",
        "district_name_key",
    )
    return (
        staged
        .join(
            F.broadcast(canonical_districts),
            staged["district_name_key_source"]
            == canonical_districts["district_name_key"],
            "left",
        )
        .withColumn(
            "district_mapping_status",
            F.when(
                F.col("district_name_source").isNull(),
                F.lit("source_district_missing"),
            )
            .when(
                F.col("district_code").isNull(),
                F.lit("unmatched_source_district"),
            )
            .otherwise(F.lit("canonical_match")),
        )
        .select(
            "cell_id",
            "source_feature_id",
            "district_code",
            "district_number",
            "district_name",
            "district_name_key",
            "district_name_source",
            "district_name_key_source",
            "district_mapping_status",
            *GRID_FEATURE_COLUMNS,
            "geometry_wkb",
            "geometry_type",
            "geometry_crs",
            "geometry_crs_source",
            "source_layer",
            *TECHNICAL_METADATA_COLUMNS,
        )
    )


def validate_grid_features(
    bronze_grid: DataFrame,
    gold_grid: DataFrame,
    *,
    expected_record_count: int = 10_100,
    expected_source_district_missing_rows: int = 433,
) -> dict[str, int]:
    """Validate identity, district coverage, feature ranges, and geometry."""

    bronze_summary = bronze_grid.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("cell_id").alias("distinct_cell_ids"),
    ).first().asDict()
    expected_source_summary = {
        "row_count": expected_record_count,
        "distinct_cell_ids": expected_record_count,
    }
    if bronze_summary != expected_source_summary:
        raise ValueError(
            "Bronze grid does not satisfy the expected snapshot contract: "
            f"expected={expected_source_summary}, found={bronze_summary}"
        )

    summary = gold_grid.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("cell_id").alias("distinct_cell_ids"),
        F.countDistinct("source_feature_id").alias("distinct_source_feature_ids"),
        F.countDistinct("district_code").alias("district_count"),
        F.sum(
            (F.col("district_mapping_status") == "canonical_match").cast("int")
        ).alias("canonical_district_matches"),
        F.sum(
            (
                F.col("district_mapping_status")
                == "source_district_missing"
            ).cast("int")
        ).alias("source_district_missing_rows"),
        F.sum(
            (
                F.col("district_mapping_status")
                == "unmatched_source_district"
            ).cast("int")
        ).alias("unmatched_source_district_rows"),
        F.sum(
            (
                (
                    (F.col("district_mapping_status") == "canonical_match")
                    & F.col("district_code").isNull()
                )
                | (
                    (F.col("district_mapping_status") != "canonical_match")
                    & F.col("district_code").isNotNull()
                )
                | (
                    (F.col("district_mapping_status") == "source_district_missing")
                    & F.col("district_name_source").isNotNull()
                )
                | (
                    (F.col("district_mapping_status") == "unmatched_source_district")
                    & F.col("district_name_source").isNull()
                )
            ).cast("int")
        ).alias("inconsistent_district_mapping_rows"),
        F.sum(
            (
                F.col("land_use_code").isNull()
                | F.col("land_use_class").isNull()
                | F.col("land_use_class_simplified").isNull()
            ).cast("int")
        ).alias("missing_land_use_rows"),
        F.sum(
            (
                (F.col("building_height_mean_m") < 0)
                | (F.col("building_height_max_m") < 0)
                | (
                    F.col("building_height_mean_m").isNotNull()
                    & F.col("building_height_max_m").isNotNull()
                    & (
                        F.col("building_height_max_m")
                        < F.col("building_height_mean_m")
                    )
                )
            ).cast("int")
        ).alias("invalid_height_rows"),
        F.sum((F.col("public_transport_stop_count") < 0).cast("int")).alias(
            "invalid_transport_rows"
        ),
        F.sum((F.col("rent_median_eur_m2_2023") <= 0).cast("int")).alias(
            "invalid_rent_rows"
        ),
        F.sum((F.col("emvs_housing_units_total") < 0).cast("int")).alias(
            "invalid_emvs_rows"
        ),
        F.sum(
            (
                F.col("geometry_wkb").isNull()
                | (F.length("geometry_wkb") == 0)
            ).cast("int")
        ).alias("empty_geometry_rows"),
        F.sum(
            (
                F.col("geometry_type").isNull()
                | ~F.col("geometry_type").isin("Polygon", "MultiPolygon")
            ).cast("int")
        ).alias("invalid_geometry_type_rows"),
        F.sum(
            (
                F.col("geometry_crs").isNull()
                | (F.col("geometry_crs") != "EPSG:3035")
            ).cast("int")
        ).alias("crs_failures"),
        F.sum(
            (
                F.col("_source_kind").isNull()
                | (F.col("_source_kind") != "derived_snapshot")
            ).cast("int")
        ).alias("source_kind_failures"),
    ).first().asDict()
    expected_summary = {
        "row_count": expected_record_count,
        "distinct_cell_ids": expected_record_count,
        "distinct_source_feature_ids": expected_record_count,
        "district_count": 21,
        "canonical_district_matches": (
            expected_record_count - expected_source_district_missing_rows
        ),
        "source_district_missing_rows": expected_source_district_missing_rows,
        "unmatched_source_district_rows": 0,
        "inconsistent_district_mapping_rows": 0,
        "missing_land_use_rows": 0,
        "invalid_height_rows": 0,
        "invalid_transport_rows": 0,
        "invalid_rent_rows": 0,
        "invalid_emvs_rows": 0,
        "empty_geometry_rows": 0,
        "invalid_geometry_type_rows": 0,
        "crs_failures": 0,
        "source_kind_failures": 0,
    }
    failures = {
        key: (expected_value, summary.get(key))
        for key, expected_value in expected_summary.items()
        if summary.get(key) != expected_value
    }
    if failures:
        raise ValueError(f"Gold grid-feature validation failed: {failures}")
    return {key: int(value) for key, value in summary.items()}


def grid_feature_null_profile(gold_grid: DataFrame) -> dict[str, int]:
    """Report missing analytical inputs without applying an imputation policy."""

    return {
        key: int(value)
        for key, value in gold_grid.agg(
            *[
                F.sum(F.col(column_name).isNull().cast("int")).alias(column_name)
                for column_name in GRID_FEATURE_COLUMNS
            ]
        ).first().asDict().items()
    }


def build_latest_district_indicators(indicator_panel: DataFrame) -> DataFrame:
    """Select each target indicator's newest district observation deterministically."""

    target_names = list(DISTRICT_INDICATOR_FEATURES.values())
    district_rows = indicator_panel.filter(
        (F.col("geographic_level") == "district")
        & F.col("indicator_name").isin(*target_names)
    )
    ordering = Window.partitionBy("district_code", "indicator_name").orderBy(
        F.col("observation_year").desc(),
        F.col("panel_year").desc(),
        F.col("observation_date").desc(),
        F.col("source_record_id").desc(),
    )
    latest = district_rows.withColumn(
        "_indicator_selection_rank",
        F.row_number().over(ordering),
    ).filter(F.col("_indicator_selection_rank") == 1)

    result = latest.select("district_code").distinct()
    for feature_name, indicator_name in DISTRICT_INDICATOR_FEATURES.items():
        selected = latest.filter(F.col("indicator_name") == indicator_name).select(
            "district_code",
            F.col("indicator_value").cast("double").alias(feature_name),
            F.col("observation_year").alias(f"{feature_name}_year"),
            F.col("panel_year").alias(f"{feature_name}_panel_year"),
            F.col("source_record_id").alias(f"{feature_name}_source_record_id"),
            F.col("observation_key").alias(f"{feature_name}_observation_key"),
            F.col("indicator_source").alias(f"{feature_name}_source_label"),
            F.col("has_conflicting_observations").alias(
                f"{feature_name}_has_conflict"
            ),
        )
        result = result.join(selected, on="district_code", how="left")

    indicator_lineage = district_rows.groupBy("district_code").agg(
        F.first("_source_id", ignorenulls=True).alias("indicator_source_id"),
        F.first("_source_sha256", ignorenulls=True).alias(
            "indicator_source_sha256"
        ),
        F.first("_bronze_run_id", ignorenulls=True).alias(
            "indicator_bronze_run_id"
        ),
        F.first("_silver_run_id", ignorenulls=True).alias(
            "indicator_silver_run_id"
        ),
    )
    return result.join(indicator_lineage, on="district_code", how="left")


def build_district_features(
    district_reference: DataFrame,
    district_boundaries: DataFrame,
    population: DataFrame,
    emvs_housing: DataFrame,
    indicator_panel: DataFrame,
    gold_grid: DataFrame,
) -> DataFrame:
    """Build one analytical feature row for each canonical Madrid district."""

    canonical_districts = district_reference.select(
        "district_code",
        "district_number",
        "district_name",
        "district_name_key",
    )
    boundaries = district_boundaries.select(
        "district_code",
        "area_m2",
        (F.col("area_m2") / F.lit(1_000_000.0)).alias("area_km2"),
        "analysis_geometry_wkb",
        "analysis_geometry_type",
        "analysis_geometry_crs",
        F.col("_source_id").alias("boundary_source_id"),
        F.col("_source_sha256").alias("boundary_source_sha256"),
        F.col("_bronze_run_id").alias("boundary_bronze_run_id"),
        F.col("_silver_run_id").alias("boundary_silver_run_id"),
    )
    latest_population = (
        population
        .filter(
            (F.col("geographic_level") == "district")
            & (F.col("reference_year") == 2024)
        )
        .select(
            "district_code",
            F.col("reference_date").alias("population_reference_date"),
            F.col("reference_year").alias("population_reference_year"),
            "population_total",
            "population_male",
            "population_female",
            F.col("_source_id").alias("population_source_id"),
            F.col("_source_sha256").alias("population_source_sha256"),
            F.col("_bronze_run_id").alias("population_bronze_run_id"),
            F.col("_silver_run_id").alias("population_silver_run_id"),
        )
    )
    housing = emvs_housing.select(
        "district_code",
        F.col("reporting_start_date").alias("housing_reporting_start_date"),
        F.col("reporting_end_date").alias("housing_reporting_end_date"),
        "housing_units_total",
        "housing_units_regulation",
        "housing_units_other_programs",
        F.col("_source_id").alias("housing_source_id"),
        F.col("_source_sha256").alias("housing_source_sha256"),
        F.col("_bronze_run_id").alias("housing_bronze_run_id"),
        F.col("_silver_run_id").alias("housing_silver_run_id"),
    )
    latest_indicators = build_latest_district_indicators(indicator_panel)
    mapped_grid = gold_grid.filter(
        F.col("district_mapping_status") == "canonical_match"
    )
    grid_aggregates = mapped_grid.groupBy("district_code").agg(
        F.count("*").alias("grid_cell_count"),
        F.count("building_height_mean_m").alias(
            "grid_height_observed_cell_count"
        ),
        F.avg("building_height_mean_m").alias(
            "grid_building_height_mean_m_avg"
        ),
        F.avg("building_height_max_m").alias(
            "grid_building_height_max_m_avg"
        ),
        F.avg(F.col("public_transport_stop_count").cast("double")).alias(
            "grid_public_transport_stop_count_avg"
        ),
        F.avg(
            F.col("has_good_public_transport_access").cast("double")
        ).alias("grid_good_public_transport_access_share"),
        F.avg(
            F.col("land_use_class_simplified")
            .isin(*GREEN_LIKE_CLASSES)
            .cast("double")
        ).alias("grid_green_like_share"),
        F.avg(
            F.col("land_use_class_simplified")
            .isin(*RESIDENTIAL_CLASSES)
            .cast("double")
        ).alias("grid_residential_share"),
        F.avg(
            F.col("land_use_class_simplified")
            .isin(*INDUSTRIAL_CLASSES)
            .cast("double")
        ).alias("grid_industrial_share"),
        F.avg(
            F.col("land_use_class_simplified")
            .isin(*DENSE_URBAN_CLASSES)
            .cast("double")
        ).alias("grid_dense_urban_share"),
        F.count("rent_median_eur_m2_2023").alias(
            "grid_rent_observed_cell_count"
        ),
        F.avg("rent_median_eur_m2_2023").alias(
            "grid_rent_median_eur_m2_2023"
        ),
        F.first("_source_id", ignorenulls=True).alias("grid_source_id"),
        F.first("_source_sha256", ignorenulls=True).alias("grid_source_sha256"),
        F.first("_bronze_run_id", ignorenulls=True).alias("grid_bronze_run_id"),
        F.first("_gold_run_id", ignorenulls=True).alias("grid_gold_run_id"),
    )
    grid_aggregates = (
        grid_aggregates
        .withColumn(
            "grid_height_coverage_share",
            F.col("grid_height_observed_cell_count") / F.col("grid_cell_count"),
        )
        .withColumn(
            "grid_rent_coverage_share",
            F.col("grid_rent_observed_cell_count") / F.col("grid_cell_count"),
        )
    )

    return (
        canonical_districts
        .join(boundaries, on="district_code", how="left")
        .join(latest_population, on="district_code", how="left")
        .join(housing, on="district_code", how="left")
        .join(latest_indicators, on="district_code", how="left")
        .join(grid_aggregates, on="district_code", how="left")
        .withColumn(
            "population_density_per_km2",
            F.round(F.col("population_total") / F.col("area_km2"), 0),
        )
        .withColumn(
            "housing_units_per_1000_residents",
            F.round(
                F.col("housing_units_total")
                / F.col("population_total")
                * F.lit(1_000.0),
                2,
            ),
        )
        .withColumn(
            "has_green_data",
            F.col("green_area_per_10000").isNotNull(),
        )
        .withColumn(
            "has_economy_data",
            F.col("income_per_person").isNotNull()
            & F.col("household_income").isNotNull(),
        )
        .withColumn(
            "has_employment_data",
            F.col("unemployment_rate").isNotNull(),
        )
        .withColumn(
            "has_vulnerability_data",
            F.col("vulnerability_index").isNotNull()
            & F.col("vulnerability_employment").isNotNull(),
        )
        .withColumn(
            "district_indicator_coverage_share",
            sum(
                F.col(feature_name).isNotNull().cast("double")
                for feature_name in DISTRICT_INDICATOR_FEATURES
            )
            / F.lit(float(len(DISTRICT_INDICATOR_FEATURES))),
        )
        .withColumn(
            "district_feature_coverage_score",
            sum(
                F.col(column_name).isNotNull().cast("double")
                for column_name in (
                    "population_total",
                    "population_density_per_km2",
                    "housing_units_total",
                    "housing_units_per_1000_residents",
                    "green_area_per_10000",
                    "income_per_person",
                    "household_income",
                    "unemployment_rate",
                    "vulnerability_index",
                    "vulnerability_employment",
                    "grid_building_height_mean_m_avg",
                    "grid_public_transport_stop_count_avg",
                    "grid_green_like_share",
                    "grid_dense_urban_share",
                )
            )
            / F.lit(14.0),
        )
    )


def validate_district_features(
    gold_grid: DataFrame,
    district_features: DataFrame,
) -> dict[str, int]:
    """Validate district coverage, measures, aggregations, and lineage."""

    mapped_grid = gold_grid.filter(
        F.col("district_mapping_status") == "canonical_match"
    )
    grid_input_summary = mapped_grid.agg(
        F.count("*").alias("mapped_grid_rows"),
        F.countDistinct("district_code").alias("mapped_grid_districts"),
        F.count("building_height_mean_m").alias("height_observed_rows"),
        F.count("rent_median_eur_m2_2023").alias("rent_observed_rows"),
        F.sum(
            (
                F.col("building_height_mean_m").isNull()
                != F.col("building_height_max_m").isNull()
            ).cast("int")
        ).alias("height_pair_mismatch_rows"),
    ).first().asDict()
    expected_grid_input = {
        "mapped_grid_rows": 9_667,
        "mapped_grid_districts": 21,
        "height_pair_mismatch_rows": 0,
    }
    grid_input_failures = {
        key: (expected_value, grid_input_summary.get(key))
        for key, expected_value in expected_grid_input.items()
        if grid_input_summary.get(key) != expected_value
    }
    inconsistent_rent_districts = (
        mapped_grid
        .groupBy("district_code")
        .agg(
            F.countDistinct("rent_median_eur_m2_2023").alias(
                "distinct_rent_values"
            )
        )
        .filter(F.col("distinct_rent_values") > 1)
        .count()
    )
    if inconsistent_rent_districts:
        grid_input_failures["inconsistent_rent_districts"] = (
            0,
            inconsistent_rent_districts,
        )

    summary = district_features.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("district_code").alias("distinct_district_codes"),
        F.sum(
            (
                F.col("population_reference_date").isNull()
                | F.col("population_reference_year").isNull()
                | (F.col("population_reference_year") != 2024)
            ).cast("int")
        ).alias("population_year_failures"),
        F.sum(
            (
                F.col("population_total").isNull()
                | F.col("population_male").isNull()
                | F.col("population_female").isNull()
                | (
                    F.col("population_total")
                    != F.col("population_male") + F.col("population_female")
                )
            ).cast("int")
        ).alias("population_failures"),
        F.sum(
            (
                F.col("area_m2").isNull()
                | (F.col("area_m2") <= 0)
                | F.col("population_density_per_km2").isNull()
            ).cast("int")
        ).alias("area_density_failures"),
        F.sum(
            (
                F.col("analysis_geometry_wkb").isNull()
                | (F.length("analysis_geometry_wkb") == 0)
                | F.col("analysis_geometry_type").isNull()
                | ~F.col("analysis_geometry_type").isin(
                    "Polygon", "MultiPolygon"
                )
                | F.col("analysis_geometry_crs").isNull()
                | (F.col("analysis_geometry_crs") != "EPSG:3035")
            ).cast("int")
        ).alias("boundary_geometry_failures"),
        F.sum(
            (
                F.col("housing_reporting_start_date").isNull()
                | F.col("housing_reporting_end_date").isNull()
                | (
                    F.col("housing_reporting_end_date")
                    < F.col("housing_reporting_start_date")
                )
                |
                F.col("housing_units_total").isNull()
                | F.col("housing_units_regulation").isNull()
                | F.col("housing_units_per_1000_residents").isNull()
            ).cast("int")
        ).alias("housing_failures"),
        F.sum(F.col("grid_cell_count")).alias("aggregated_grid_cells"),
        F.sum(
            (
                F.col("grid_cell_count").isNull()
                | (F.col("grid_cell_count") <= 0)
                | (F.col("grid_height_observed_cell_count") < 0)
                | (
                    F.col("grid_height_observed_cell_count")
                    > F.col("grid_cell_count")
                )
                | (F.col("grid_rent_observed_cell_count") < 0)
                | (
                    F.col("grid_rent_observed_cell_count")
                    > F.col("grid_cell_count")
                )
            ).cast("int")
        ).alias("grid_count_failures"),
        F.sum("grid_height_observed_cell_count").alias(
            "aggregated_height_observed_cells"
        ),
        F.sum("grid_rent_observed_cell_count").alias(
            "aggregated_rent_observed_cells"
        ),
        F.sum(
            (
                (
                    (F.col("grid_height_observed_cell_count") > 0)
                    & (
                        F.col("grid_building_height_mean_m_avg").isNull()
                        | F.col("grid_building_height_max_m_avg").isNull()
                    )
                )
                | (
                    (F.col("grid_rent_observed_cell_count") > 0)
                    & F.col("grid_rent_median_eur_m2_2023").isNull()
                )
            ).cast("int")
        ).alias("observed_value_failures"),
        F.sum(
            (
                ~F.col("grid_height_coverage_share").between(0.0, 1.0)
                | ~F.col("grid_rent_coverage_share").between(0.0, 1.0)
                | ~F.col("grid_good_public_transport_access_share").between(
                    0.0, 1.0
                )
                | ~F.col("grid_green_like_share").between(0.0, 1.0)
                | ~F.col("grid_residential_share").between(0.0, 1.0)
                | ~F.col("grid_industrial_share").between(0.0, 1.0)
                | ~F.col("grid_dense_urban_share").between(0.0, 1.0)
                | ~F.col("district_indicator_coverage_share").between(0.0, 1.0)
                | ~F.col("district_feature_coverage_score").between(0.0, 1.0)
            ).cast("int")
        ).alias("share_range_failures"),
        F.sum(
            (
                sum(
                    (
                        F.col(feature_name).isNotNull()
                        & (
                            F.col(f"{feature_name}_year").isNull()
                            | ~F.col(f"{feature_name}_year").between(2008, 2025)
                            | F.col(
                                f"{feature_name}_source_record_id"
                            ).isNull()
                            | F.col(f"{feature_name}_observation_key").isNull()
                            | F.coalesce(
                                F.col(f"{feature_name}_has_conflict"),
                                F.lit(False),
                            )
                        )
                    ).cast("int")
                    for feature_name in DISTRICT_INDICATOR_FEATURES
                )
                > 0
            ).cast("int")
        ).alias("indicator_selection_failure_rows"),
        *[
            F.count(feature_name).alias(f"{feature_name}_district_count")
            for feature_name in DISTRICT_INDICATOR_FEATURES
        ],
        F.sum(
            (
                F.col("boundary_source_id").isNull()
                | F.col("population_source_id").isNull()
                | F.col("housing_source_id").isNull()
                | F.col("grid_source_id").isNull()
                | F.col("boundary_source_sha256").isNull()
                | F.col("population_source_sha256").isNull()
                | F.col("housing_source_sha256").isNull()
                | F.col("grid_source_sha256").isNull()
                | F.col("indicator_source_id").isNull()
                | F.col("indicator_source_sha256").isNull()
                | F.col("boundary_silver_run_id").isNull()
                | F.col("boundary_bronze_run_id").isNull()
                | F.col("population_silver_run_id").isNull()
                | F.col("population_bronze_run_id").isNull()
                | F.col("housing_silver_run_id").isNull()
                | F.col("housing_bronze_run_id").isNull()
                | F.col("grid_gold_run_id").isNull()
                | F.col("grid_bronze_run_id").isNull()
                | F.col("indicator_silver_run_id").isNull()
                | F.col("indicator_bronze_run_id").isNull()
            ).cast("int")
        ).alias("lineage_failures"),
    ).first().asDict()
    expected_summary = {
        "row_count": 21,
        "distinct_district_codes": 21,
        "population_year_failures": 0,
        "population_failures": 0,
        "area_density_failures": 0,
        "boundary_geometry_failures": 0,
        "housing_failures": 0,
        "aggregated_grid_cells": 9_667,
        "grid_count_failures": 0,
        "observed_value_failures": 0,
        "share_range_failures": 0,
        "indicator_selection_failure_rows": 0,
        "lineage_failures": 0,
    }
    failures = {
        key: (expected_value, summary.get(key))
        for key, expected_value in expected_summary.items()
        if summary.get(key) != expected_value
    }
    observation_reconciliation = {
        "aggregated_height_observed_cells": (
            grid_input_summary["height_observed_rows"],
            summary["aggregated_height_observed_cells"],
        ),
        "aggregated_rent_observed_cells": (
            grid_input_summary["rent_observed_rows"],
            summary["aggregated_rent_observed_cells"],
        ),
    }
    for key, (expected_value, actual_value) in observation_reconciliation.items():
        if actual_value != expected_value:
            failures[key] = (expected_value, actual_value)
    for feature_name in DISTRICT_INDICATOR_FEATURES:
        count_key = f"{feature_name}_district_count"
        if not summary[count_key]:
            failures[count_key] = ("> 0", summary[count_key])
    if grid_input_failures or failures:
        raise ValueError(
            "Gold district-feature validation failed: "
            f"grid_input={grid_input_failures}, district={failures}"
        )
    return {
        **{key: int(value) for key, value in grid_input_summary.items()},
        **{key: int(value) for key, value in summary.items()},
    }


def write_snapshot_table(frame: DataFrame, table_name: str) -> None:
    """Replace one deterministic Gold snapshot and its schema."""

    (
        frame.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )


def validate_persisted_gold_table(
    spark: SparkSession,
    *,
    table_name: str,
    expected_record_count: int,
) -> DataFrame:
    """Read a Gold table back and enforce its schema and lineage contract."""

    persisted = spark.table(table_name)
    persisted_count = persisted.count()
    if persisted_count != expected_record_count:
        raise ValueError(
            f"Persisted count mismatch for {table_name}: expected "
            f"{expected_record_count}, found {persisted_count}."
        )
    required_columns = [
        "cell_id",
        "source_feature_id",
        "district_code",
        "district_name",
        "district_mapping_status",
        *GRID_FEATURE_COLUMNS,
        "geometry_wkb",
        "geometry_type",
        "geometry_crs",
        *TECHNICAL_METADATA_COLUMNS,
        *GOLD_METADATA_COLUMNS,
    ]
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in persisted.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Persisted table {table_name} is missing columns: {missing_columns}"
        )
    incomplete_lineage_rows = persisted.filter(
        F.col("_source_id").isNull()
        | F.col("_source_sha256").isNull()
        | F.col("_bronze_run_id").isNull()
        | F.col("_gold_run_id").isNull()
        | F.col("_gold_processed_at_utc").isNull()
    ).count()
    if incomplete_lineage_rows:
        raise ValueError(
            f"Persisted table {table_name} has {incomplete_lineage_rows} rows "
            "without complete Gold lineage."
        )
    return persisted


def validate_persisted_district_feature_table(
    spark: SparkSession,
    *,
    table_name: str,
    expected_record_count: int = 21,
) -> DataFrame:
    """Read district features back and enforce schema and Gold metadata."""

    persisted = spark.table(table_name)
    persisted_count = persisted.count()
    if persisted_count != expected_record_count:
        raise ValueError(
            f"Persisted count mismatch for {table_name}: expected "
            f"{expected_record_count}, found {persisted_count}."
        )
    required_columns = [
        "district_code",
        "district_number",
        "district_name",
        "district_name_key",
        *DISTRICT_FEATURE_COLUMNS,
        "analysis_geometry_wkb",
        "analysis_geometry_crs",
        "boundary_source_id",
        "population_source_id",
        "housing_source_id",
        "grid_source_id",
        "indicator_source_id",
        "boundary_silver_run_id",
        "population_silver_run_id",
        "housing_silver_run_id",
        "grid_gold_run_id",
        "indicator_silver_run_id",
        *[
            column_name
            for feature_name in DISTRICT_INDICATOR_FEATURES
            for column_name in (
                f"{feature_name}_panel_year",
                f"{feature_name}_source_record_id",
                f"{feature_name}_observation_key",
                f"{feature_name}_source_label",
                f"{feature_name}_has_conflict",
            )
        ],
        *GOLD_METADATA_COLUMNS,
    ]
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in persisted.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Persisted table {table_name} is missing columns: {missing_columns}"
        )
    incomplete_gold_metadata = persisted.filter(
        F.col("_gold_run_id").isNull()
        | F.col("_gold_processed_at_utc").isNull()
    ).count()
    if incomplete_gold_metadata:
        raise ValueError(
            f"Persisted table {table_name} has {incomplete_gold_metadata} rows "
            "without Gold run metadata."
        )
    return persisted
