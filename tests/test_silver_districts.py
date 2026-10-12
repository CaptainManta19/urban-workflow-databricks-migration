import unittest

from urban_workflow.silver.districts import (
    canonicalize_emvs_district_key,
    normalize_district_name,
)


class SilverDistrictNormalizationTests(unittest.TestCase):
    def test_normalizes_accents_spacing_and_punctuation(self) -> None:
        self.assertEqual(normalize_district_name("  Fuencarral - El Pardo "), "FUENCARRALELPARDO")
        self.assertEqual(normalize_district_name("Vicálvaro"), "VICALVARO")
        self.assertEqual(normalize_district_name("Tetuán"), "TETUAN")

    def test_maps_the_three_emvs_legacy_aliases(self) -> None:
        expected = {
            "MONCLOA": "MONCLOAARAVACA",
            "SAN BLAS": "SANBLASCANILLEJAS",
            "VALLECAS VILLA": "VILLADEVALLECAS",
        }
        for source_name, expected_key in expected.items():
            with self.subTest(source_name=source_name):
                key, method = canonicalize_emvs_district_key(source_name)
                self.assertEqual(key, expected_key)
                self.assertEqual(method, "explicit_alias")

    def test_uses_normalized_name_for_direct_matches(self) -> None:
        key, method = canonicalize_emvs_district_key("CHAMBERI")

        self.assertEqual(key, "CHAMBERI")
        self.assertEqual(method, "normalized_name")


if __name__ == "__main__":
    unittest.main()
