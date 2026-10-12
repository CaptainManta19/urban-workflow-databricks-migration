"""Reproducible KMeans urban-typology modeling for the Gold grid."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLUSTER_COUNT = 3
RANDOM_STATE = 42
N_INIT = 20
CATEGORICAL_FEATURES = ("land_use_class_simplified",)
NUMERIC_FEATURES = (
    "building_height_mean_m",
    "building_height_max_m",
    "public_transport_stop_count",
)


@dataclass
class KMeansArtifacts:
    """Fitted objects and tabular outputs from one typology run."""

    preprocessor: ColumnTransformer
    model: KMeans
    feature_names: list[str]
    grid_predictions: pd.DataFrame
    cluster_profiles: pd.DataFrame
    district_cluster_mix: pd.DataFrame
    eligible_count: int
    silhouette: float


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _ineligibility_reason(row: pd.Series) -> str | None:
    if row["district_mapping_status"] != "canonical_match":
        return str(row["district_mapping_status"])
    missing = [
        column_name
        for column_name in (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
        if pd.isna(row[column_name])
    ]
    return ";".join(f"missing_{name}" for name in missing) or None


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "land_use",
                Pipeline([("encoder", _one_hot_encoder())]),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "numeric",
                Pipeline([("scaler", StandardScaler())]),
                list(NUMERIC_FEATURES),
            ),
        ]
    )


def _balance_feature_groups(
    feature_matrix: np.ndarray,
    *,
    land_use_feature_count: int,
) -> np.ndarray:
    balanced = np.asarray(feature_matrix, dtype=float).copy()
    numeric_feature_count = len(NUMERIC_FEATURES)
    balanced[:, :land_use_feature_count] /= math.sqrt(land_use_feature_count)
    balanced[
        :, land_use_feature_count : land_use_feature_count + numeric_feature_count
    ] /= math.sqrt(numeric_feature_count)
    return balanced


def _narrative_label(profile: pd.Series) -> str:
    land_use = profile["dominant_land_use_class"]
    mean_transport = profile["mean_public_transport_stop_count"]
    if land_use in {
        "Continuous urban fabric (S.L. : > 80%)",
        "Discontinuous dense urban fabric (S.L. : 50% -  80%)",
    }:
        return (
            "Dense accessible urban fabric"
            if mean_transport >= 4
            else "Dense urban fabric"
        )
    if land_use == "Green urban areas":
        return "Green urban fabric"
    return "Mixed urban fabric"


def fit_kmeans_typology(grid_features: pd.DataFrame) -> KMeansArtifacts:
    """Fit the legacy-compatible three-cluster model and build its outputs."""

    required_columns = {
        "cell_id",
        "district_code",
        "district_name",
        "district_mapping_status",
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    }
    missing_columns = sorted(required_columns - set(grid_features.columns))
    if missing_columns:
        raise ValueError(f"Gold grid is missing clustering columns: {missing_columns}")
    if grid_features["cell_id"].duplicated().any():
        raise ValueError("Gold grid contains duplicate cell IDs.")

    labeled = grid_features.copy()
    labeled["cluster_ineligibility_reason"] = labeled.apply(
        _ineligibility_reason,
        axis=1,
    )
    labeled["cluster_eligible"] = labeled["cluster_ineligibility_reason"].isna()
    eligible = labeled.loc[labeled["cluster_eligible"]].copy()
    if len(eligible) < CLUSTER_COUNT:
        raise ValueError("Too few eligible grid cells for three-cluster KMeans.")

    preprocessor = _build_preprocessor()
    transformed = preprocessor.fit_transform(
        eligible[[*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]]
    )
    encoder = preprocessor.named_transformers_["land_use"].named_steps["encoder"]
    categorical_names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = [*categorical_names, *NUMERIC_FEATURES]
    balanced = _balance_feature_groups(
        transformed,
        land_use_feature_count=len(categorical_names),
    )

    model = KMeans(
        n_clusters=CLUSTER_COUNT,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    labels = model.fit_predict(balanced)
    distances = np.linalg.norm(balanced - model.cluster_centers_[labels], axis=1)
    eligible["cluster_label"] = [f"cluster_{value}" for value in labels]
    eligible["cluster_distance_to_centroid"] = distances
    labeled = labeled.merge(
        eligible[["cell_id", "cluster_label", "cluster_distance_to_centroid"]],
        on="cell_id",
        how="left",
    )

    profile_rows: list[dict[str, object]] = []
    for cluster_label, cluster_frame in eligible.groupby("cluster_label"):
        dominant_land_use = str(
            cluster_frame["land_use_class_simplified"].mode().iloc[0]
        )
        profile_rows.append(
            {
                "cluster_label": cluster_label,
                "cell_count": int(len(cluster_frame)),
                "share_of_eligible_cells": float(len(cluster_frame) / len(eligible)),
                "dominant_land_use_class": dominant_land_use,
                "mean_building_height_mean_m": float(
                    cluster_frame["building_height_mean_m"].mean()
                ),
                "mean_building_height_max_m": float(
                    cluster_frame["building_height_max_m"].mean()
                ),
                "mean_public_transport_stop_count": float(
                    cluster_frame["public_transport_stop_count"].mean()
                ),
            }
        )
    profiles = pd.DataFrame(profile_rows).sort_values("cluster_label").reset_index(
        drop=True
    )
    profiles["narrative_label"] = profiles.apply(_narrative_label, axis=1)

    mix = (
        pd.crosstab(eligible["district_code"], eligible["cluster_label"])
        .reindex(columns=[f"cluster_{index}" for index in range(CLUSTER_COUNT)], fill_value=0)
    )
    mix = mix.div(mix.sum(axis=1), axis=0).add_prefix("cluster_share_").reset_index()
    district_names = eligible[["district_code", "district_name"]].drop_duplicates()
    mix = district_names.merge(mix, on="district_code", how="left")
    share_columns = [f"cluster_share_cluster_{index}" for index in range(CLUSTER_COUNT)]
    mix["dominant_cluster_label"] = (
        mix[share_columns].idxmax(axis=1).str.removeprefix("cluster_share_")
    )

    silhouette = float(silhouette_score(balanced, labels))
    return KMeansArtifacts(
        preprocessor=preprocessor,
        model=model,
        feature_names=feature_names,
        grid_predictions=labeled[
            [
                "cell_id",
                "cluster_eligible",
                "cluster_ineligibility_reason",
                "cluster_label",
                "cluster_distance_to_centroid",
            ]
        ],
        cluster_profiles=profiles,
        district_cluster_mix=mix,
        eligible_count=int(len(eligible)),
        silhouette=silhouette,
    )
