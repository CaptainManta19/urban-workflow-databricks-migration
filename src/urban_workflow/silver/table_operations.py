"""Spark transformations and acceptance checks for Silver snapshot tables."""

from datetime import datetime

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BinaryType,
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from urban_workflow.bronze.table_operations import TECHNICAL_METADATA_COLUMNS
from urban_workflow.silver.districts import EMVS_DISTRICT_ALIASES
from urban_workflow.silver.geospatial_operations import (
    DistrictBoundaryConversion,
    repair_district_boundary_records,
)
from urban_workflow.silver.population import CITY_AGGREGATE_CODE


SILVER_METADATA_COLUMNS = (
    "_silver_run_id",
    "_silver_processed_at_utc",
)
EMVS_REPORTING_START_DATE = "2015-06-01"
EMVS_REPORTING_END_DATE = "2023-04-30"


def normalized_district_key(column: Column) -> Column:
    """Build the Spark equivalent of the canonical Python normalization rule."""

    return F.regexp_replace(
        F.translate(F.upper(F.trim(column)), "ÁÉÍÓÚÜÑ", "AEIOUUN"),
        "[^A-Z0-9]",
        "",
    )


def _emvs_canonical_key(column: Column) -> Column:
    alias_entries = []
    for source_key, canonical_key in EMVS_DISTRICT_ALIASES.items():
        alias_entries.extend([F.lit(source_key), F.lit(canonical_key)])
    alias_map = F.create_map(*alias_entries)
    normalized_key = normalized_district_key(column)
    return F.coalesce(F.element_at(alias_map, normalized_key), normalized_key)


def add_silver_metadata(
    frame: DataFrame,
    *,
    silver_run_id: str,
    processed_at_utc: datetime,
) -> DataFrame:
    """Attach the shared processing identity used by Silver snapshots."""

    return (
        frame
        .withColumn("_silver_run_id", F.lit(silver_run_id))
        .withColumn("_silver_processed_at_utc", F.lit(processed_at_utc))
    )


def build_district_reference(boundaries: DataFrame) -> DataFrame:
    """Create the authoritative, non-geometric Madrid district lookup."""

    lineage_columns = [F.col(column_name) for column_name in TECHNICAL_METADATA_COLUMNS]
    return boundaries.select(
        F.col("COD_DIS_TX").alias("district_code"),
        F.col("COD_DIS").cast("int").alias("district_number"),
        F.col("NOMBRE").alias("district_name"),
        normalized_district_key(F.col("NOMBRE")).alias("district_name_key"),
        F.col("DISTRI_MAY").alias("district_name_upper_source"),
        F.col("DISTRI_MT").alias("district_name_display_source"),
        *lineage_columns,
    )


def build_emvs_housing(
    bronze_emvs: DataFrame,
    district_reference: DataFrame,
) -> DataFrame:
    """Type EMVS measures and map every source label to a canonical district."""

    source_rows = (
        bronze_emvs
        .filter(F.col("DISTRITOS") != "TOTAL")
        .withColumn("district_name_source", F.col("DISTRITOS"))
        .withColumn(
            "district_name_key_source",
            normalized_district_key(F.col("DISTRITOS")),
        )
        .withColumn(
            "district_name_key",
            _emvs_canonical_key(F.col("DISTRITOS")),
        )
        .withColumn(
            "district_mapping_method",
            F.when(
                F.col("district_name_key_source") != F.col("district_name_key"),
                F.lit("explicit_alias"),
            ).otherwise(F.lit("normalized_name")),
        )
        .withColumn("housing_units_total", F.expr("try_cast(TOTAL AS BIGINT)"))
        .withColumn(
            "housing_units_regulation",
            F.expr("try_cast(REGLAMENTO_ADJUDICACION AS BIGINT)"),
        )
        .withColumn(
            "housing_units_other_programs",
            F.expr("try_cast(RESTO_PROGRAMAS AS BIGINT)"),
        )
        .withColumn(
            "reporting_start_date",
            F.to_date(F.lit(EMVS_REPORTING_START_DATE)),
        )
        .withColumn(
            "reporting_end_date",
            F.to_date(F.lit(EMVS_REPORTING_END_DATE)),
        )
    )

    reference_columns = district_reference.select(
        "district_code",
        "district_number",
        "district_name",
        "district_name_key",
    )
    return (
        source_rows
        .join(F.broadcast(reference_columns), on="district_name_key", how="left")
        .select(
            "district_code",
            "district_number",
            "district_name",
            "district_name_key",
            "district_name_source",
            "district_name_key_source",
            "district_mapping_method",
            "reporting_start_date",
            "reporting_end_date",
            "housing_units_total",
            "housing_units_regulation",
            "housing_units_other_programs",
            *TECHNICAL_METADATA_COLUMNS,
        )
    )


def validate_district_reference(frame: DataFrame) -> dict[str, int]:
    """Enforce the 21-row canonical Madrid district contract."""

    summary = frame.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("district_code").alias("distinct_district_codes"),
        F.countDistinct("district_name_key").alias("distinct_name_keys"),
        F.sum(
            (
                F.col("district_code").isNull()
                | F.col("district_name").isNull()
                | F.col("district_name_key").isNull()
            ).cast("int")
        ).alias("missing_required_values"),
    ).first().asDict()
    expected_codes = {f"{district_number:02d}" for district_number in range(1, 22)}
    actual_codes = {
        row["district_code"]
        for row in frame.select("district_code").collect()
    }
    if summary != {
        "row_count": 21,
        "distinct_district_codes": 21,
        "distinct_name_keys": 21,
        "missing_required_values": 0,
    } or actual_codes != expected_codes:
        raise ValueError(
            "District reference does not satisfy the canonical 21-district contract: "
            f"summary={summary}, codes={sorted(actual_codes)}"
        )
    return {key: int(value) for key, value in summary.items()}


def validate_emvs_housing(
    bronze_emvs: DataFrame,
    silver_emvs: DataFrame,
) -> dict[str, int]:
    """Reconcile typed district rows to the preserved Bronze TOTAL record."""

    bronze_summary = bronze_emvs.agg(
        F.count("*").alias("row_count"),
        F.sum((F.col("DISTRITOS") == "TOTAL").cast("int")).alias("total_rows"),
        F.sum(F.col("DISTRITOS").isNull().cast("int")).alias("missing_district_names"),
        F.sum(
            (
                F.coalesce(F.length(F.trim(F.col("_UNNAMED_5"))), F.lit(0)) > 0
            ).cast("int")
        ).alias("unexpected_position_5_values"),
        F.sum(
            (
                F.coalesce(F.length(F.trim(F.col("_UNNAMED_6"))), F.lit(0)) > 0
            ).cast("int")
        ).alias("unexpected_position_6_values"),
    ).first().asDict()
    expected_bronze_summary = {
        "row_count": 22,
        "total_rows": 1,
        "missing_district_names": 0,
        "unexpected_position_5_values": 0,
        "unexpected_position_6_values": 0,
    }
    if bronze_summary != expected_bronze_summary:
        raise ValueError(
            "Bronze EMVS does not satisfy the expected source-shape contract: "
            f"{bronze_summary}"
        )

    total_rows = bronze_emvs.filter(F.col("DISTRITOS") == "TOTAL")
    expected = total_rows.select(
        F.expr("try_cast(TOTAL AS BIGINT)").alias("housing_units_total"),
        F.expr("try_cast(REGLAMENTO_ADJUDICACION AS BIGINT)").alias(
            "housing_units_regulation"
        ),
        F.expr("try_cast(RESTO_PROGRAMAS AS BIGINT)").alias(
            "housing_units_other_programs"
        ),
    ).first().asDict()
    if any(value is None for value in expected.values()):
        raise ValueError(f"Bronze EMVS TOTAL values are not valid integers: {expected}")

    summary = silver_emvs.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("district_code").alias("distinct_district_codes"),
        F.sum(F.col("district_code").isNull().cast("int")).alias(
            "unmatched_districts"
        ),
        F.sum(F.col("housing_units_total")).alias("housing_units_total"),
        F.sum(F.col("housing_units_regulation")).alias(
            "housing_units_regulation"
        ),
        F.sum(F.col("housing_units_other_programs")).alias(
            "housing_units_other_programs"
        ),
        F.sum(F.col("housing_units_total").isNull().cast("int")).alias(
            "total_parse_failures"
        ),
        F.sum(F.col("housing_units_regulation").isNull().cast("int")).alias(
            "regulation_parse_failures"
        ),
        F.sum(F.col("housing_units_other_programs").isNull().cast("int")).alias(
            "missing_other_programs"
        ),
        F.sum(
            (F.col("district_mapping_method") == "explicit_alias").cast("int")
        ).alias("explicit_alias_rows"),
        F.sum(
            (
                F.col("housing_units_other_programs").isNotNull()
                & (
                    F.col("housing_units_total")
                    != F.col("housing_units_regulation")
                    + F.col("housing_units_other_programs")
                )
            ).cast("int")
        ).alias("row_reconciliation_failures"),
        F.sum(
            (
                F.col("housing_units_other_programs").isNull()
                & (
                    F.col("housing_units_total")
                    != F.col("housing_units_regulation")
                )
            ).cast("int")
        ).alias("missing_other_programs_nonzero_gap"),
    ).first().asDict()

    required_summary = {
        "row_count": 21,
        "distinct_district_codes": 21,
        "unmatched_districts": 0,
        "total_parse_failures": 0,
        "regulation_parse_failures": 0,
        "missing_other_programs": 2,
        "explicit_alias_rows": 3,
        "row_reconciliation_failures": 0,
        "missing_other_programs_nonzero_gap": 0,
    }
    failures = {
        key: (required_value, summary.get(key))
        for key, required_value in required_summary.items()
        if summary.get(key) != required_value
    }
    total_failures = {
        key: (expected[key], summary.get(key))
        for key in expected
        if summary.get(key) != expected[key]
    }
    if failures or total_failures:
        raise ValueError(
            "Silver EMVS validation failed: "
            f"quality={failures}, reconciliation={total_failures}"
        )
    return {key: int(value) for key, value in summary.items()}


def build_population(
    bronze_population: DataFrame,
    district_reference: DataFrame,
) -> DataFrame:
    """Type the complete population history and normalize its geographic grain."""

    geographic_level = (
        F.when(F.col("cod_distrito") == CITY_AGGREGATE_CODE, F.lit("city"))
        .when(F.col("cod_distrito") == F.col("cod_barrio"), F.lit("district"))
        .otherwise(F.lit("barrio"))
    )
    staged = (
        bronze_population
        .withColumn("source_record_id", F.col("_id"))
        .withColumn("reference_date_source", F.col("fecha"))
        .withColumn(
            "reference_year",
            F.regexp_extract(
                F.col("fecha"),
                r"^1 de enero de ((?:19|20)\d{2})$",
                1,
            ).cast("int"),
        )
        .withColumn(
            "reference_date",
            F.make_date(F.col("reference_year"), F.lit(1), F.lit(1)),
        )
        .withColumn("geographic_level", geographic_level)
        .withColumn("municipality_code", F.col("cod_municipio"))
        .withColumn("municipality_name", F.col("municipio"))
        .withColumn(
            "district_code",
            F.when(
                F.col("geographic_level") != "city",
                F.lpad(F.col("cod_distrito"), 2, "0"),
            ),
        )
        .withColumn("district_name_source", F.col("distrito"))
        .withColumn(
            "barrio_code",
            F.when(
                F.col("geographic_level") == "barrio",
                F.lpad(F.col("cod_barrio"), 3, "0"),
            ),
        )
        .withColumn("barrio_name_source", F.col("barrio"))
        .withColumn(
            "barrio_name",
            F.when(F.col("geographic_level") == "barrio", F.col("barrio")),
        )
        .withColumn(
            "population_total",
            F.expr("try_cast(regexp_replace(num_personas, '[.]', '') AS BIGINT)"),
        )
        .withColumn(
            "population_male",
            F.expr(
                "try_cast(regexp_replace(num_personas_hombres, '[.]', '') AS BIGINT)"
            ),
        )
        .withColumn(
            "population_female",
            F.expr(
                "try_cast(regexp_replace(num_personas_mujeres, '[.]', '') AS BIGINT)"
            ),
        )
    )
    reference_columns = district_reference.select(
        "district_code",
        "district_name",
        "district_name_key",
    )
    return (
        staged
        .join(F.broadcast(reference_columns), on="district_code", how="left")
        .select(
            "source_record_id",
            "reference_date_source",
            "reference_date",
            "reference_year",
            "geographic_level",
            "municipality_code",
            "municipality_name",
            "district_code",
            "district_name",
            "district_name_key",
            "district_name_source",
            "barrio_code",
            "barrio_name",
            "barrio_name_source",
            "population_total",
            "population_male",
            "population_female",
            *TECHNICAL_METADATA_COLUMNS,
        )
    )


def validate_population(
    bronze_population: DataFrame,
    silver_population: DataFrame,
) -> dict[str, int]:
    """Validate population typing, keys, coverage, and geographic rollups."""

    bronze_summary = bronze_population.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("_id").alias("distinct_source_record_ids"),
    ).first().asDict()
    if bronze_summary != {
        "row_count": 1_071,
        "distinct_source_record_ids": 1_071,
    }:
        raise ValueError(
            "Bronze population does not satisfy the expected snapshot contract: "
            f"{bronze_summary}"
        )

    natural_key = F.struct(
        "reference_date",
        "geographic_level",
        "municipality_code",
        "district_code",
        "barrio_code",
    )
    summary = silver_population.agg(
        F.count("*").alias("row_count"),
        F.countDistinct(natural_key).alias("distinct_natural_keys"),
        F.countDistinct("reference_date").alias("reference_date_count"),
        F.min("reference_year").alias("minimum_reference_year"),
        F.max("reference_year").alias("maximum_reference_year"),
        F.sum((F.col("geographic_level") == "city").cast("int")).alias(
            "city_rows"
        ),
        F.sum((F.col("geographic_level") == "district").cast("int")).alias(
            "district_rows"
        ),
        F.sum((F.col("geographic_level") == "barrio").cast("int")).alias(
            "barrio_rows"
        ),
        F.sum(
            (
                F.col("reference_date").isNull()
                | F.col("population_total").isNull()
                | F.col("population_male").isNull()
                | F.col("population_female").isNull()
            ).cast("int")
        ).alias("parse_failures"),
        F.sum(
            (
                F.col("population_total")
                != F.col("population_male") + F.col("population_female")
            ).cast("int")
        ).alias("sex_reconciliation_failures"),
        F.sum(
            (
                (F.col("geographic_level") != "city")
                & F.col("district_name").isNull()
            ).cast("int")
        ).alias("district_join_failures"),
        F.sum(
            (
                (F.col("geographic_level") == "barrio")
                & F.col("barrio_code").isNull()
            ).cast("int")
        ).alias("missing_barrio_codes"),
    ).first().asDict()

    expected_summary = {
        "row_count": 1_071,
        "distinct_natural_keys": 1_071,
        "reference_date_count": 7,
        "minimum_reference_year": 2018,
        "maximum_reference_year": 2024,
        "city_rows": 7,
        "district_rows": 147,
        "barrio_rows": 917,
        "parse_failures": 0,
        "sex_reconciliation_failures": 0,
        "district_join_failures": 0,
        "missing_barrio_codes": 0,
    }
    summary_failures = {
        key: (expected_value, summary.get(key))
        for key, expected_value in expected_summary.items()
        if summary.get(key) != expected_value
    }

    city_totals = silver_population.filter(
        F.col("geographic_level") == "city"
    ).select(
        "reference_date",
        F.col("population_total").alias("city_population"),
    )
    district_totals = (
        silver_population
        .filter(F.col("geographic_level") == "district")
        .groupBy("reference_date")
        .agg(F.sum("population_total").alias("district_population"))
    )
    city_district_reconciliation_failures = (
        city_totals
        .join(district_totals, on="reference_date", how="full")
        .filter(
            F.col("city_population").isNull()
            | F.col("district_population").isNull()
            | (F.col("city_population") != F.col("district_population"))
        )
        .count()
    )

    district_records = silver_population.filter(
        F.col("geographic_level") == "district"
    ).select(
        "reference_date",
        "district_code",
        F.col("population_total").alias("district_population"),
    )
    barrio_totals = (
        silver_population
        .filter(F.col("geographic_level") == "barrio")
        .groupBy("reference_date", "district_code")
        .agg(F.sum("population_total").alias("barrio_population"))
    )
    district_barrio_reconciliation = district_records.join(
        barrio_totals,
        on=["reference_date", "district_code"],
        how="full",
    )
    district_barrio_reconciliation_rows = district_barrio_reconciliation.count()
    district_barrio_reconciliation_failures = district_barrio_reconciliation.filter(
        F.col("district_population").isNull()
        | F.col("barrio_population").isNull()
        | (F.col("district_population") != F.col("barrio_population"))
    ).count()

    reconciliation_summary = {
        "city_district_reconciliation_failures": (
            city_district_reconciliation_failures
        ),
        "district_barrio_reconciliation_rows": (
            district_barrio_reconciliation_rows
        ),
        "district_barrio_reconciliation_failures": (
            district_barrio_reconciliation_failures
        ),
    }
    expected_reconciliation = {
        "city_district_reconciliation_failures": 0,
        "district_barrio_reconciliation_rows": 147,
        "district_barrio_reconciliation_failures": 0,
    }
    reconciliation_failures = {
        key: (expected_value, reconciliation_summary[key])
        for key, expected_value in expected_reconciliation.items()
        if reconciliation_summary[key] != expected_value
    }
    if summary_failures or reconciliation_failures:
        raise ValueError(
            "Silver population validation failed: "
            f"quality={summary_failures}, reconciliation={reconciliation_failures}"
        )

    return {
        **{key: int(value) for key, value in summary.items()},
        **reconciliation_summary,
    }


def build_indicator_panel(
    bronze_indicators: DataFrame,
    district_reference: DataFrame,
) -> DataFrame:
    """Type the long indicator history and annotate conflicting observations."""

    geographic_level = (
        F.when(F.col("cod_distrito").isNull(), F.lit("city"))
        .when(F.col("cod_barrio").isNull(), F.lit("district"))
        .otherwise(F.lit("barrio"))
    )
    staged = (
        bronze_indicators
        .withColumn("source_record_id", F.col("_id"))
        .withColumn("source_order", F.expr("try_cast(Orden AS BIGINT)"))
        .withColumn("panel_year", F.expr("try_cast(periodo_panel AS INT)"))
        .withColumn("observation_year", F.expr("try_cast(ano AS INT)"))
        .withColumn(
            "observation_date",
            F.to_date(F.col("fecha_indicador"), "dd/MM/yyyy"),
        )
        .withColumn("geographic_level", geographic_level)
        .withColumn("city_name", F.col("ciudad"))
        .withColumn(
            "district_code",
            F.when(
                F.col("geographic_level") != "city",
                F.lpad(F.col("cod_distrito"), 2, "0"),
            ),
        )
        .withColumn("district_name_source", F.col("distrito"))
        .withColumn(
            "barrio_code",
            F.when(
                F.col("geographic_level") == "barrio",
                F.lpad(F.col("cod_barrio"), 3, "0"),
            ),
        )
        .withColumn("barrio_name", F.col("barrio"))
        .withColumn("indicator_category_1", F.col("categoria_1"))
        .withColumn("indicator_category_2", F.col("categoria_2"))
        .withColumn("indicator_level_1", F.col("indicador_nivel1"))
        .withColumn("indicator_level_2", F.col("indicador_nivel2"))
        .withColumn("indicator_level_3", F.col("indicador_nivel3"))
        .withColumn("indicator_unit", F.col("unidad_indicador"))
        .withColumn("indicator_name", F.col("indicador_completo"))
        .withColumn("indicator_source", F.col("fuente_indicador"))
        .withColumn("indicator_value_raw", F.col("valor_indicador"))
        .withColumn(
            "indicator_value",
            F.expr(
                "try_cast(replace(replace(valor_indicador, '.', ''), ',', '.') "
                "AS DECIMAL(18,9))"
            ),
        )
    )
    reference_columns = district_reference.select(
        "district_code",
        "district_name",
        "district_name_key",
    )
    staged = staged.join(
        F.broadcast(reference_columns),
        on="district_code",
        how="left",
    )

    observation_identity_columns = (
        "panel_year",
        "observation_year",
        "observation_date",
        "geographic_level",
        "city_name",
        "district_code",
        "barrio_code",
        "indicator_source",
        "indicator_category_1",
        "indicator_category_2",
        "indicator_level_1",
        "indicator_level_2",
        "indicator_level_3",
        "indicator_unit",
        "indicator_name",
    )
    staged = staged.withColumn(
        "observation_key",
        F.sha2(
            F.to_json(
                F.struct(
                    *[
                        F.col(column_name).alias(column_name)
                        for column_name in observation_identity_columns
                    ]
                ),
                options={"ignoreNullFields": "false"},
            ),
            256,
        ),
    )
    observation_quality = staged.groupBy("observation_key").agg(
        F.count("*").alias("observation_group_count"),
        F.countDistinct("indicator_value").alias("conflicting_value_count"),
    )
    return (
        staged
        .join(observation_quality, on="observation_key", how="left")
        .withColumn(
            "has_conflicting_observations",
            F.col("conflicting_value_count") > 1,
        )
        .select(
            "source_record_id",
            "source_order",
            "observation_key",
            "panel_year",
            "observation_year",
            "observation_date",
            "geographic_level",
            "city_name",
            "district_code",
            "district_name",
            "district_name_key",
            "district_name_source",
            "barrio_code",
            "barrio_name",
            "indicator_category_1",
            "indicator_category_2",
            "indicator_level_1",
            "indicator_level_2",
            "indicator_level_3",
            "indicator_unit",
            "indicator_name",
            "indicator_source",
            "indicator_value_raw",
            "indicator_value",
            "observation_group_count",
            "conflicting_value_count",
            "has_conflicting_observations",
            *TECHNICAL_METADATA_COLUMNS,
        )
    )


def validate_indicator_panel(
    bronze_indicators: DataFrame,
    silver_indicators: DataFrame,
) -> dict[str, int]:
    """Validate indicator typing, geographic coverage, and conflict annotation."""

    bronze_summary = bronze_indicators.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("_id").alias("distinct_source_record_ids"),
    ).first().asDict()
    if bronze_summary != {
        "row_count": 116_930,
        "distinct_source_record_ids": 116_930,
    }:
        raise ValueError(
            "Bronze indicators do not satisfy the expected snapshot contract: "
            f"{bronze_summary}"
        )

    summary = silver_indicators.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("source_record_id").alias("distinct_source_record_ids"),
        F.countDistinct("observation_key").alias("distinct_observation_keys"),
        F.countDistinct("panel_year").alias("panel_year_count"),
        F.min("panel_year").alias("minimum_panel_year"),
        F.max("panel_year").alias("maximum_panel_year"),
        F.countDistinct("observation_year").alias("observation_year_count"),
        F.min("observation_year").alias("minimum_observation_year"),
        F.max("observation_year").alias("maximum_observation_year"),
        F.sum((F.col("geographic_level") == "city").cast("int")).alias(
            "city_rows"
        ),
        F.sum((F.col("geographic_level") == "district").cast("int")).alias(
            "district_rows"
        ),
        F.sum((F.col("geographic_level") == "barrio").cast("int")).alias(
            "barrio_rows"
        ),
        F.sum(
            (
                F.col("source_order").isNull()
                | F.col("panel_year").isNull()
                | F.col("observation_year").isNull()
                | F.col("observation_date").isNull()
                | F.col("indicator_value").isNull()
            ).cast("int")
        ).alias("parse_failures"),
        F.sum(
            (
                (F.col("geographic_level") != "city")
                & F.col("district_name").isNull()
            ).cast("int")
        ).alias("district_join_failures"),
        F.sum(
            (
                (F.col("geographic_level") == "barrio")
                & F.col("barrio_code").isNull()
            ).cast("int")
        ).alias("missing_barrio_codes"),
        F.countDistinct(
            F.when(
                F.col("has_conflicting_observations"),
                F.col("observation_key"),
            )
        ).alias("conflict_group_count"),
        F.sum(F.col("has_conflicting_observations").cast("int")).alias(
            "conflict_rows"
        ),
        F.max(
            F.when(
                F.col("has_conflicting_observations"),
                F.col("observation_group_count"),
            )
        ).alias("maximum_conflict_group_size"),
        F.sum(
            (
                F.col("has_conflicting_observations")
                & (
                    (F.col("observation_group_count") != 2)
                    | (F.col("conflicting_value_count") != 2)
                )
            ).cast("int")
        ).alias("invalid_conflict_annotations"),
    ).first().asDict()

    expected_summary = {
        "row_count": 116_930,
        "distinct_source_record_ids": 116_930,
        "distinct_observation_keys": 116_922,
        "panel_year_count": 6,
        "minimum_panel_year": 2020,
        "maximum_panel_year": 2025,
        "observation_year_count": 18,
        "minimum_observation_year": 2008,
        "maximum_observation_year": 2025,
        "city_rows": 1_804,
        "district_rows": 30_556,
        "barrio_rows": 84_570,
        "parse_failures": 0,
        "district_join_failures": 0,
        "missing_barrio_codes": 0,
        "conflict_group_count": 8,
        "conflict_rows": 16,
        "maximum_conflict_group_size": 2,
        "invalid_conflict_annotations": 0,
    }
    failures = {
        key: (expected_value, summary.get(key))
        for key, expected_value in expected_summary.items()
        if summary.get(key) != expected_value
    }
    if failures:
        raise ValueError(f"Silver indicator validation failed: {failures}")
    return {key: int(value) for key, value in summary.items()}


def build_district_boundaries(
    spark: SparkSession,
    bronze_boundaries: DataFrame,
) -> tuple[DataFrame, DistrictBoundaryConversion]:
    """Repair the driver-sized boundary set and construct an explicit Spark frame."""

    source_records = [
        row.asDict(recursive=True)
        for row in bronze_boundaries.collect()
    ]
    conversion = repair_district_boundary_records(source_records)
    schema = StructType(
        [
            StructField("district_code", StringType(), nullable=False),
            StructField("district_number", IntegerType(), nullable=False),
            StructField("district_name", StringType(), nullable=False),
            StructField("district_name_upper_source", StringType(), nullable=True),
            StructField("district_name_display_source", StringType(), nullable=True),
            StructField("source_geometry_wkb", BinaryType(), nullable=False),
            StructField("source_geometry_type", StringType(), nullable=False),
            StructField("source_geometry_crs", StringType(), nullable=False),
            StructField("source_geometry_crs_source", StringType(), nullable=False),
            StructField("source_geometry_was_valid", BooleanType(), nullable=False),
            StructField(
                "source_geometry_validity_reason",
                StringType(),
                nullable=True,
            ),
            StructField("geometry_repaired", BooleanType(), nullable=False),
            StructField("geometry_repair_method", StringType(), nullable=False),
            StructField("geometry_wkb", BinaryType(), nullable=False),
            StructField("geometry_type", StringType(), nullable=False),
            StructField("geometry_crs", StringType(), nullable=False),
            StructField("geometry_is_valid", BooleanType(), nullable=False),
            StructField("analysis_geometry_wkb", BinaryType(), nullable=False),
            StructField("analysis_geometry_type", StringType(), nullable=False),
            StructField("analysis_geometry_crs", StringType(), nullable=False),
            StructField("analysis_geometry_is_valid", BooleanType(), nullable=False),
            StructField("source_area_m2", DoubleType(), nullable=False),
            StructField("area_m2", DoubleType(), nullable=False),
            StructField(
                "repair_area_relative_change",
                DoubleType(),
                nullable=True,
            ),
            StructField("source_layer", StringType(), nullable=False),
            StructField("_source_id", StringType(), nullable=False),
            StructField("_source_kind", StringType(), nullable=False),
            StructField("_collection_method", StringType(), nullable=False),
            StructField("_source_path", StringType(), nullable=False),
            StructField("_source_sha256", StringType(), nullable=False),
            StructField("_bronze_run_id", StringType(), nullable=False),
            StructField("_ingested_at_utc", TimestampType(), nullable=False),
        ]
    )
    return spark.createDataFrame(conversion.records, schema=schema), conversion


def validate_district_boundaries(
    boundaries: DataFrame,
    district_reference: DataFrame,
) -> dict[str, int | float]:
    """Enforce geometry validity, repair evidence, area stability, and coverage."""

    summary = boundaries.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("district_code").alias("distinct_district_codes"),
        F.sum((~F.col("source_geometry_was_valid")).cast("int")).alias(
            "source_invalid_rows"
        ),
        F.sum(F.col("geometry_repaired").cast("int")).alias("repaired_rows"),
        F.sum((~F.col("geometry_is_valid")).cast("int")).alias(
            "invalid_mapping_geometries"
        ),
        F.sum((~F.col("analysis_geometry_is_valid")).cast("int")).alias(
            "invalid_analysis_geometries"
        ),
        F.sum((F.col("geometry_type") == "Polygon").cast("int")).alias(
            "polygon_rows"
        ),
        F.sum((F.col("geometry_type") == "MultiPolygon").cast("int")).alias(
            "multipolygon_rows"
        ),
        F.sum(
            (
                F.col("geometry_wkb").isNull()
                | (F.length("geometry_wkb") == 0)
                | F.col("analysis_geometry_wkb").isNull()
                | (F.length("analysis_geometry_wkb") == 0)
            ).cast("int")
        ).alias("empty_geometry_rows"),
        F.sum(
            (
                (F.col("geometry_crs") != "EPSG:4326")
                | (F.col("analysis_geometry_crs") != "EPSG:3035")
            ).cast("int")
        ).alias("crs_failures"),
        F.sum((F.col("area_m2") <= 0).cast("int")).alias("nonpositive_area_rows"),
        F.max(F.abs("repair_area_relative_change")).alias(
            "maximum_absolute_relative_area_change"
        ),
    ).first().asDict()
    reference_join_failures = (
        boundaries.select("district_code")
        .join(
            district_reference.select("district_code"),
            on="district_code",
            how="left_anti",
        )
        .count()
    )
    expected_counts = {
        "row_count": 21,
        "distinct_district_codes": 21,
        "source_invalid_rows": 3,
        "repaired_rows": 3,
        "invalid_mapping_geometries": 0,
        "invalid_analysis_geometries": 0,
        "polygon_rows": 20,
        "multipolygon_rows": 1,
        "empty_geometry_rows": 0,
        "crs_failures": 0,
        "nonpositive_area_rows": 0,
    }
    failures = {
        key: (expected_value, summary.get(key))
        for key, expected_value in expected_counts.items()
        if summary.get(key) != expected_value
    }
    maximum_area_change = summary.get("maximum_absolute_relative_area_change")
    if maximum_area_change is None or maximum_area_change > 0.000001:
        failures["maximum_absolute_relative_area_change"] = (
            "<= 0.000001",
            maximum_area_change,
        )
    if reference_join_failures:
        failures["reference_join_failures"] = (0, reference_join_failures)
    if failures:
        raise ValueError(f"Silver district-boundary validation failed: {failures}")
    return {
        **{
            key: int(value)
            for key, value in summary.items()
            if key != "maximum_absolute_relative_area_change"
        },
        "maximum_absolute_relative_area_change": float(maximum_area_change),
        "reference_join_failures": reference_join_failures,
    }


def write_snapshot_table(frame: DataFrame, table_name: str) -> None:
    """Replace one deterministic Silver snapshot and its schema."""

    (
        frame.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )


def validate_persisted_silver_table(
    spark: SparkSession,
    *,
    table_name: str,
    expected_record_count: int,
    required_columns: list[str],
) -> DataFrame:
    """Read a Silver table back and enforce row, schema, and run metadata checks."""

    persisted = spark.table(table_name)
    if persisted.count() != expected_record_count:
        raise ValueError(
            f"Persisted count mismatch for {table_name}: expected "
            f"{expected_record_count}, found {persisted.count()}."
        )
    missing_columns = [
        column_name
        for column_name in [
            *required_columns,
            *TECHNICAL_METADATA_COLUMNS,
            *SILVER_METADATA_COLUMNS,
        ]
        if column_name not in persisted.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Persisted table {table_name} is missing columns: {missing_columns}"
        )
    metadata_nulls = persisted.filter(
        F.col("_silver_run_id").isNull()
        | F.col("_silver_processed_at_utc").isNull()
        | F.col("_source_id").isNull()
        | F.col("_source_sha256").isNull()
        | F.col("_bronze_run_id").isNull()
    ).count()
    if metadata_nulls:
        raise ValueError(
            f"Persisted table {table_name} has {metadata_nulls} rows without Silver metadata."
        )
    return persisted
