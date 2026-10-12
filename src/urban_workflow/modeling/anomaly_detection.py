"""District-level IsolationForest mismatch detection and explanations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CONTAMINATION = 0.10
RANDOM_STATE = 42
N_ESTIMATORS = 300
BASE_FEATURE_COLUMNS = (
    "population_density_per_km2",
    "housing_units_per_1000_residents",
    "green_area_per_10000",
    "income_per_person",
    "household_income",
    "unemployment_rate",
    "vulnerability_index",
    "vulnerability_employment",
    "grid_good_public_transport_access_share",
    "grid_building_height_mean_m_avg",
    "grid_green_like_share",
    "grid_dense_urban_share",
)
CLUSTER_SHARE_COLUMNS = tuple(
    f"cluster_share_cluster_{index}" for index in range(3)
)
ANOMALY_FEATURE_COLUMNS = (*BASE_FEATURE_COLUMNS, *CLUSTER_SHARE_COLUMNS)


@dataclass
class IsolationForestArtifacts:
    """Fitted objects and district-level anomaly outputs."""

    preprocessor: Pipeline
    model: IsolationForest
    feature_columns: list[str]
    district_predictions: pd.DataFrame
    eligible_count: int
    flagged_count: int


def _interpretation_notes(
    scaled_row: pd.Series,
    top_features: list[str],
) -> list[str]:
    notes = []
    for feature_name in top_features:
        direction = "higher" if scaled_row[feature_name] > 0 else "lower"
        notes.append(
            f"{feature_name} is unusually {direction} than the district average pattern."
        )
    return notes


def fit_isolation_forest(
    district_features: pd.DataFrame,
    district_cluster_mix: pd.DataFrame,
) -> IsolationForestArtifacts:
    """Fit the reviewed district mismatch model and create ranked explanations."""

    required_district_columns = {
        "district_code",
        "district_name",
        *BASE_FEATURE_COLUMNS,
    }
    required_mix_columns = {
        "district_code",
        *CLUSTER_SHARE_COLUMNS,
    }
    missing_district = sorted(
        required_district_columns - set(district_features.columns)
    )
    missing_mix = sorted(required_mix_columns - set(district_cluster_mix.columns))
    if missing_district or missing_mix:
        raise ValueError(
            "IsolationForest inputs are incomplete: "
            f"district={missing_district}, cluster_mix={missing_mix}"
        )
    if district_features["district_code"].duplicated().any():
        raise ValueError("District features contain duplicate district codes.")
    if district_cluster_mix["district_code"].duplicated().any():
        raise ValueError("District cluster mix contains duplicate district codes.")

    merged = district_features.merge(
        district_cluster_mix[["district_code", *CLUSTER_SHARE_COLUMNS]],
        on="district_code",
        how="left",
        validate="one_to_one",
    )
    eligible_mask = merged[list(ANOMALY_FEATURE_COLUMNS)].notna().any(axis=1)
    eligible = merged.loc[eligible_mask].copy()
    if eligible.empty:
        raise ValueError("No districts are eligible for IsolationForest.")

    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    feature_matrix = preprocessor.fit_transform(
        eligible[list(ANOMALY_FEATURE_COLUMNS)]
    )
    model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS,
    )
    model.fit(feature_matrix)
    scores = -model.score_samples(feature_matrix)
    flags = model.predict(feature_matrix) == -1
    scaled = pd.DataFrame(
        feature_matrix,
        index=eligible.index,
        columns=ANOMALY_FEATURE_COLUMNS,
    )

    predictions = merged[["district_code", "district_name"]].copy()
    predictions["anomaly_eligible"] = eligible_mask
    predictions["anomaly_score"] = np.nan
    predictions["anomaly_flag"] = None
    predictions["anomaly_rank"] = None
    predictions["anomaly_top_features"] = [[] for _ in range(len(predictions))]
    predictions["anomaly_interpretation_notes"] = [
        [] for _ in range(len(predictions))
    ]
    predictions.loc[eligible.index, "anomaly_score"] = scores
    predictions.loc[eligible.index, "anomaly_flag"] = flags
    ranked_indices = eligible.index[np.argsort(-scores)]
    for rank, row_index in enumerate(ranked_indices, start=1):
        predictions.at[row_index, "anomaly_rank"] = rank
        standardized_row = scaled.loc[row_index]
        top_features = list(
            standardized_row.abs().sort_values(ascending=False).head(3).index
        )
        predictions.at[row_index, "anomaly_top_features"] = top_features
        predictions.at[row_index, "anomaly_interpretation_notes"] = (
            _interpretation_notes(standardized_row, top_features)
        )

    return IsolationForestArtifacts(
        preprocessor=preprocessor,
        model=model,
        feature_columns=list(ANOMALY_FEATURE_COLUMNS),
        district_predictions=predictions,
        eligible_count=int(len(eligible)),
        flagged_count=int(flags.sum()),
    )
