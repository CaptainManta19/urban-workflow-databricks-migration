import unittest


try:
    import pandas as pd

    from urban_workflow.modeling.anomaly_detection import fit_isolation_forest
    from urban_workflow.modeling.clustering import (
        CLUSTER_COUNT,
        fit_kmeans_typology,
    )

    MODELING_DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    MODELING_DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(
    MODELING_DEPENDENCIES_AVAILABLE,
    "Pandas, NumPy, and scikit-learn are required for clustering tests.",
)
class KMeansClusteringTests(unittest.TestCase):
    def test_preserves_ineligible_rows_and_builds_complete_outputs(self) -> None:
        rows = []
        for district_index, district_code in enumerate(("01", "02", "03")):
            for cell_offset in range(4):
                rows.append(
                    {
                        "cell_id": district_index * 10 + cell_offset,
                        "district_code": district_code,
                        "district_name": f"District {district_code}",
                        "district_mapping_status": "canonical_match",
                        "land_use_class_simplified": (
                            "Green urban areas"
                            if district_index == 0
                            else f"Land use {district_index}"
                        ),
                        "building_height_mean_m": 5.0 + district_index * 20 + cell_offset,
                        "building_height_max_m": 8.0 + district_index * 25 + cell_offset,
                        "public_transport_stop_count": district_index * 5 + cell_offset,
                    }
                )
        rows.append(
            {
                "cell_id": 99,
                "district_code": None,
                "district_name": None,
                "district_mapping_status": "source_district_missing",
                "land_use_class_simplified": "Green urban areas",
                "building_height_mean_m": None,
                "building_height_max_m": None,
                "public_transport_stop_count": 0,
            }
        )

        result = fit_kmeans_typology(pd.DataFrame(rows))

        self.assertEqual(result.eligible_count, 12)
        self.assertEqual(len(result.grid_predictions), 13)
        self.assertEqual(len(result.cluster_profiles), CLUSTER_COUNT)
        self.assertEqual(len(result.district_cluster_mix), 3)
        self.assertAlmostEqual(
            float(result.cluster_profiles["share_of_eligible_cells"].sum()),
            1.0,
        )
        ineligible = result.grid_predictions.loc[
            ~result.grid_predictions["cluster_eligible"]
        ].iloc[0]
        self.assertEqual(
            ineligible["cluster_ineligibility_reason"],
            "source_district_missing",
        )
        self.assertTrue(pd.isna(ineligible["cluster_label"]))

    def test_scores_and_ranks_all_districts_for_anomaly_review(self) -> None:
        district_rows = []
        mix_rows = []
        for index in range(21):
            district_code = f"{index + 1:02d}"
            district_rows.append(
                {
                    "district_code": district_code,
                    "district_name": f"District {district_code}",
                    "population_density_per_km2": 1000.0 + index * 100,
                    "housing_units_per_1000_residents": 1.0 + index / 10,
                    "green_area_per_10000": 5.0 + index,
                    "income_per_person": 15000.0 + index * 500,
                    "household_income": 30000.0 + index * 700,
                    "unemployment_rate": 5.0 + index / 5,
                    "vulnerability_index": 10.0 + index,
                    "vulnerability_employment": 8.0 + index,
                    "grid_good_public_transport_access_share": index / 21,
                    "grid_building_height_mean_m_avg": 8.0 + index,
                    "grid_green_like_share": (20 - index) / 21,
                    "grid_dense_urban_share": index / 21,
                }
            )
            mix_rows.append(
                {
                    "district_code": district_code,
                    "cluster_share_cluster_0": 0.5,
                    "cluster_share_cluster_1": 0.3,
                    "cluster_share_cluster_2": 0.2,
                }
            )

        result = fit_isolation_forest(
            pd.DataFrame(district_rows),
            pd.DataFrame(mix_rows),
        )

        self.assertEqual(result.eligible_count, 21)
        self.assertGreaterEqual(result.flagged_count, 1)
        self.assertEqual(
            sorted(result.district_predictions["anomaly_rank"].tolist()),
            list(range(1, 22)),
        )
        self.assertTrue(
            result.district_predictions["anomaly_top_features"].apply(len).eq(3).all()
        )


if __name__ == "__main__":
    unittest.main()
