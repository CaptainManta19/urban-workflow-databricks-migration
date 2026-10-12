import unittest

from urban_workflow.silver.population import (
    classify_population_geography,
    normalize_barrio_code,
    normalize_district_code,
    parse_population_reference_year,
    parse_spanish_integer,
)


class SilverPopulationParsingTests(unittest.TestCase):
    def test_classifies_source_geographic_levels_from_codes(self) -> None:
        self.assertEqual(classify_population_geography("Todos", "Todos"), "city")
        self.assertEqual(classify_population_geography("8", "8"), "district")
        self.assertEqual(classify_population_geography("8", "81"), "barrio")

    def test_parses_reference_year_and_spanish_integer(self) -> None:
        self.assertEqual(parse_population_reference_year("1 de enero de 2024"), 2024)
        self.assertEqual(parse_spanish_integer("3.460.491"), 3_460_491)

    def test_rejects_unexpected_reference_date(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported population reference date"):
            parse_population_reference_year("2 de enero de 2024")

    def test_normalizes_codes_by_geographic_level(self) -> None:
        self.assertIsNone(normalize_district_code("Todos"))
        self.assertEqual(normalize_district_code("8"), "08")
        self.assertIsNone(normalize_barrio_code("8", "district"))
        self.assertEqual(normalize_barrio_code("81", "barrio"), "081")

    def test_rejects_invalid_numeric_values_and_codes(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid Spanish integer"):
            parse_spanish_integer("12,5")
        with self.assertRaisesRegex(ValueError, "Invalid Madrid district code"):
            normalize_district_code("22")
        with self.assertRaisesRegex(ValueError, "Invalid Madrid barrio code"):
            normalize_barrio_code("Todos", "barrio")


if __name__ == "__main__":
    unittest.main()
